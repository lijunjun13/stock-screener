"""
A股大市值线性回归分析
- BaoStock: A-share code list
- Tencent qt.gtimg.cn: real-time quotes / market cap
- Tencent web.ifzq.gtimg.cn: 后复权 daily kline (10yr, chunked)
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import baostock as bs
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)

# ── HTTP session ──────────────────────────────────────────────────────────────
_sess = requests.Session()
_sess.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.qq.com/",
    }
)

# ── TTL cache ─────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()


def _get(key: str, ttl: int):
    with _cache_lock:
        entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _set(key: str, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}


# ── Eastmoney session (separate from Tencent _sess) ──────────────────────────
_em_sess = requests.Session()
_em_sess.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://data.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
})

# ── Analyst consensus EPS (Eastmoney datacenter) ──────────────────────────────
_EM_CONSENSUS_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"

# Financial sector keywords — these stocks use P/B valuation
_FINANCIAL_KW = {"银行", "保险", "证券", "多元金融", "期货"}
_FINANCIAL_ERP = 8.0   # higher equity risk premium for financials

def _is_financial(industry_board: str) -> bool:
    return any(kw in (industry_board or "") for kw in _FINANCIAL_KW)


# Cyclical commodity sectors — DCF model is unreliable; use historical mean PE
# Keywords are matched against Eastmoney's actual industry_board strings, e.g.:
#   工业金属 (铜/铝/锌/镍)、能源金属 (锂/钴)、普钢/特钢/冶钢原料、煤炭开采、
#   炼化及贸易/石油开采、贵金属 (金/银)、航运
_CYCLICAL_KW = {
    # ── 金属 ────────────────────────────────────────────
    "工业金属",   # 铜、铝、锌、铅、镍 (天山铝业、铜陵有色、紫金矿业、云南铜业…)
    "能源金属",   # 锂、钴 (赣锋锂业、天齐锂业…)
    "贵金属",     # 金、银 (山东黄金…)
    # ── 钢铁 ────────────────────────────────────────────
    "普钢",       # 普通钢铁 (宝山钢铁、河钢股份…)
    "特钢",       # 特种钢
    "冶钢原料",   # 焦炭、铁矿石 (方大炭素…)
    "钢铁",       # 兜底 — 万一有数据源仍用此标签
    "有色金属",   # 兜底
    # ── 煤炭 ────────────────────────────────────────────
    "煤炭",       # 煤炭开采 (中国神华、兖矿能源、陕西煤业…)
    # ── 石油 ────────────────────────────────────────────
    "炼化",       # 炼化及贸易 (中国石油、中国石化…)
    "石油",       # 石油开采
    "石化",       # 兜底
    # ── 其他大宗 ────────────────────────────────────────
    "航运",
    "采矿",
    "矿业",
}

def _is_cyclical(industry_board: str) -> bool:
    return any(kw in (industry_board or "") for kw in _CYCLICAL_KW)


def _irr_3yr(price: float, annual_div: float, target_p3: float) -> float | None:
    """3-year Internal Rate of Return.

    Solves for r in:
        price = D/(1+r) + D/(1+r)^2 + (D+P3)/(1+r)^3

    More accurate than the simple formula [(P3+3D)/P0]^(1/3)-1 because it
    correctly weights dividends received in year 1 and 2 at their true time value.
    Falls back to the simple formula if the bisection fails.
    """
    if price <= 0 or target_p3 is None:
        return None
    D, P3, P0 = annual_div, target_p3, price

    def npv(r: float) -> float:
        d = 1.0 + r
        return D / d + D / d**2 + (D + P3) / d**3 - P0

    # NPV is monotonically decreasing in r for positive cash flows.
    # Find bounds that straddle zero.
    lo, hi = -0.9, 20.0          # –90 % … +2000 %
    try:
        if npv(lo) < 0:          # even at −90 %/yr the investment loses — clamp
            return round((((P3 + 3 * D) / P0) ** (1 / 3) - 1) * 100, 1)
        if npv(hi) > 0:
            hi = 100.0           # widen upper bound for extreme growth stocks
        for _ in range(80):      # 80 iterations → precision < 1e-7
            mid = (lo + hi) / 2
            if npv(mid) > 0:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2 * 100, 1)
    except Exception:
        # Fallback
        total = (P3 + 3 * D) / P0
        return round((total ** (1 / 3) - 1) * 100, 1)


def _fetch_dividend_yield(code: str) -> float | None:
    """Return annual dividend yield (%) from Eastmoney push2 API.

    Field f162 = 股息率TTM (%).  Returns None if unavailable.
    Uses a 3-day cache — dividend data changes infrequently.
    """
    cache_key = f"div_yield_v1_{code}"
    cached = _get(cache_key, ttl=86400 * 3)
    if cached is not None:
        return cached
    result = None
    try:
        market = "1" if code.startswith("6") else "0"
        r = _em_sess.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={"secid": f"{market}.{code}", "fields": "f162"},
            timeout=5,
        )
        v = (r.json().get("data") or {}).get("f162")
        if v not in (None, "-", 0, "0"):
            dy = float(v)
            # push2 percentage fields vary by endpoint; normalise to plain %
            # If the value looks like basis-points×100 (e.g. 350 for 3.5%), divide
            if dy > 50:
                dy = dy / 100
            if 0 < dy < 30:          # sanity: 0–30 %
                result = round(dy, 2)
    except Exception:
        pass
    _set(cache_key, result)
    return result


def _fetch_all_debt_ratios() -> dict[str, float]:
    """Bulk-fetch 资产负债率 (f85, %) for all A-shares via Eastmoney push2 clist.

    Paginates through the full A-share universe and returns a dict
    {code_str: debt_ratio_pct}.  Cached for 24 h so each screener run
    only hits the API once.  Falls back to {} if the endpoint is
    unavailable.
    """
    cache_key = "debt_ratios_bulk_v1"
    cached = _get(cache_key, ttl=86400)
    if cached is not None:
        return cached
    result: dict[str, float] = {}
    try:
        PAGE = 500
        page = 1
        while True:
            r = _em_sess.get(
                "https://push2.eastmoney.com/api/qt/clist/get",
                params={
                    "pn":  str(page), "pz": str(PAGE),
                    "po":  "1",       "np": "1",
                    "ut":  "bd1d9ddb04089700cf9c27f6f7426281",
                    "fltt": "2",      "invt": "2", "fid": "f3",
                    # All main-board + GEM + STAR + Beijing A-shares
                    "fs": ("m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,"
                           "m:0+t:81+s:2048"),
                    "fields": "f12,f85",
                },
                timeout=15,
            )
            data = r.json().get("data") or {}
            diff = data.get("diff") or []
            if not diff:
                break
            for item in diff:
                code = str(item.get("f12") or "")
                v    = item.get("f85")
                if code and v not in (None, "-", ""):
                    try:
                        val = float(v)
                        if 0.0 <= val <= 100.0:
                            result[code] = round(val, 1)
                    except (ValueError, TypeError):
                        pass
            total = data.get("total") or 0
            if page * PAGE >= total:
                break
            page += 1
            time.sleep(0.4)           # polite pacing
    except Exception:
        pass
    if result:
        _set(cache_key, result)
    return result


def _fetch_debt_ratio(code: str) -> float | None:
    """Return 资产负债率 (%) for a single stock from the 24-h bulk cache."""
    return _fetch_all_debt_ratios().get(code)


def _fetch_industry_board(code: str) -> str:
    """Return industry board string for any A-share.

    Primary:  PUBLISHNAME from RPT_LICO_FN_CPD — available for all A-shares,
              consistent classification used for deduplication by sector.
    Fallback: INDUSTRY_BOARD from consensus (covered stocks only).
    Cached 24 h.
    """
    cache_key = f"industry_v2_{code}"
    cached = _get(cache_key, ttl=86400)
    if cached is not None:
        return cached

    industry = ""
    # Primary: RPT_LICO_FN_CPD.PUBLISHNAME (universal coverage)
    try:
        r = _em_sess.get(
            _EM_CONSENSUS_URL,
            params={
                "reportName":  "RPT_LICO_FN_CPD",
                "columns":     "SECURITY_CODE,PUBLISHNAME",
                "filter":      f'(SECURITY_CODE="{code}")',
                "pageSize":    "1",
                "sortColumns": "REPORTDATE",
                "sortTypes":   "-1",
                "source":      "WEB",
                "client":      "WEB",
            },
            timeout=6,
        )
        rows = ((r.json().get("result") or {}).get("data")) or []
        if rows:
            industry = str(rows[0].get("PUBLISHNAME") or "")
    except Exception:
        pass

    # Fallback: consensus INDUSTRY_BOARD (already cached, free to call)
    if not industry:
        industry = str((_fetch_consensus_eps(code) or {}).get("industry_board") or "")

    _set(cache_key, industry)
    return industry


def _fetch_realtime_pb(tc_code: str) -> float | None:
    """Return P/B ratio from Tencent real-time quote (fields[46])."""
    cache_key = f"rt_pb_{tc_code}"
    cached = _get(cache_key, ttl=300)
    if cached is not None:
        return cached
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={tc_code}", timeout=5)
        for line in r.text.strip().split(";\n"):
            if not line.strip():
                continue
            try:
                fields = line.split('"')[1].split("~")
                pb = float(fields[46])
                if 0 < pb < 500:
                    _set(cache_key, pb)
                    return pb
            except (IndexError, ValueError):
                pass
    except Exception:
        pass
    return None

# ── China 10-year government bond yield ───────────────────────────────────────
# Update _CN10Y_RATE manually when the rate changes significantly.
# Current China 10Y CGB yield as of 2026-05: ~1.68%
# Source: http://www.chinamoney.com.cn or https://yield.chinabond.com.cn
_CN10Y_RATE = 1.68

def _fetch_cn10y_yield() -> float:
    """Return China 10Y CGB yield (%).
    Uses _CN10Y_RATE which should be updated manually when the rate changes.
    """
    return _CN10Y_RATE

def _fetch_consensus_eps(code: str) -> dict:
    """Return analyst consensus data (EPS forecasts + ratings + target price).

    Source: Eastmoney profitforecast page API (RPT_WEB_RESPREDICT).
    Returns dict with keys:
      forecasts: list of {year, year_mark, eps, forward_pe}
      org_num: int  (total analyst coverage)
      buy_num: int  (buy ratings)
      add_num: int  (outperform ratings)
      target_high: float|None
      target_low:  float|None
    """
    cache_key = f"consensus_v4_{code}"
    cached = _get(cache_key, ttl=3600 * 12)
    if cached is not None:
        return cached

    empty = {"forecasts": [], "org_num": None, "buy_num": None,
             "add_num": None, "target_high": None, "target_low": None}
    try:
        params = {
            "reportName": "RPT_WEB_RESPREDICT",
            "columns":    "WEB_RESPREDICT",
            "filter":     f'(SECURITY_CODE="{code}")',
            "pageSize":   5,
            "source":     "WEB",
            "client":     "WEB",
        }
        r = _em_sess.get(_EM_CONSENSUS_URL, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        datas = ((j.get("result") or {}).get("data")) or []
        if not datas:
            _set(cache_key, empty)
            return empty

        item = datas[0]  # one row per stock
        forecasts = []
        for i in range(1, 5):   # YEAR1 … YEAR4
            year = item.get(f"YEAR{i}")
            eps  = item.get(f"EPS{i}")
            mark = item.get(f"YEAR_MARK{i}", "")
            if not year or eps is None:
                continue
            forecasts.append({
                "year":      int(year),
                "year_mark": str(mark),   # "A" actual / "E" estimate
                "eps":       round(float(eps), 2),
                "forward_pe": None,       # filled in by get_stock_data with current price
            })

        def _f(k):
            v = item.get(k)
            return round(float(v), 1) if v is not None else None

        def _i(k):
            v = item.get(k)
            return int(v) if v is not None else None

        result = {
            "forecasts":     forecasts,
            "org_num":       _i("RATING_ORG_NUM"),
            "buy_num":       _i("RATING_BUY_NUM"),
            "add_num":       _i("RATING_ADD_NUM"),
            "neutral_num":   _i("RATING_NEUTRAL_NUM"),
            "reduce_num":    _i("RATING_REDUCE_NUM"),
            "target_high":   _f("DEC_AIMPRICEMAX"),
            "target_low":    _f("DEC_AIMPRICEMIN"),
            "industry_board": str(item.get("INDUSTRY_BOARD") or ""),
        }
        _set(cache_key, result)
        return result

    except Exception:
        _set(cache_key, empty)
        return empty


def _compute_valuation_metrics(code: str, tc_code: str,
                               actual_price: float | None = None) -> dict:
    """Compute pessimistic FwdPE and 基准终局 two-stage DCF fair PE.

    Pass ``actual_price`` directly (from market-cap list) to avoid an extra
    HTTP round-trip; if omitted the price is fetched from Tencent.

    Returns dict with keys ``pess_pe`` and/or ``fair_pe_base``, or {} on any
    failure (missing consensus, negative EPS, etc.).
    """
    try:
        if actual_price is None or actual_price <= 0:
            actual_price = _fetch_realtime_price(tc_code)
        if not actual_price or actual_price <= 0:
            return {}

        consensus  = _fetch_consensus_eps(code)
        industry   = str(consensus.get("industry_board") or "")
        fcs        = sorted(consensus.get("forecasts", []), key=lambda x: x["year"])
        actual_fcs = [f for f in fcs if f.get("year_mark") == "A"]
        est_fcs    = [f for f in fcs if f.get("year_mark") == "E"]

        # Always return at least the industry so the screener can filter by sector
        base: dict = {"industry_board": industry} if industry else {}

        if not actual_fcs or len(est_fcs) < 3:
            return base

        A = actual_fcs[-1]["eps"]
        B, C, D = est_fcs[0]["eps"], est_fcs[1]["eps"], est_fcs[2]["eps"]
        if not all(v > 0 for v in (A, B, C, D)):
            return base

        B_star  = (A * B) ** 0.5
        C_star  = (A * B * C) ** (1 / 3)
        D_star  = (A * B * C * D) ** 0.25
        pess_pe = actual_price * 4 / (A + B_star + C_star + D_star)

        rf    = _fetch_cn10y_yield()
        r_dec = (rf + 5.0) / 100                        # r = rf + 5 % ERP
        ratios = [B_star / A, C_star / A, D_star / A]
        pv_s1  = sum(er / (1 + r_dec) ** i
                     for i, er in enumerate(ratios, start=1))
        disc3  = (1 + r_dec) ** 3

        g_t = rf / 100                                   # 基准终局: g_t = rf %
        base["pess_pe"] = round(pess_pe, 1)
        if g_t >= r_dec:
            return base

        tv           = ratios[-1] * (1 + g_t) / (r_dec - g_t) / disc3
        base["fair_pe_base"] = round(pv_s1 + tv, 1)

        # 基准终局 3-year pure capital gain: (P3/P0)^(1/3) − 1
        # Under Modigliani-Miller, P3 = D★·(1+g)/(r−g) already capitalises
        # the full earnings power regardless of payout ratio, so dividends
        # must NOT be added separately (that would double-count the payout).
        # Dividend yield is stored separately in the screener row for display.
        try:
            target_p3  = D_star * (1 + g_t) / (r_dec - g_t)
            if target_p3 > 0 and actual_price > 0:
                cagr = round(((target_p3 / actual_price) ** (1 / 3) - 1) * 100, 1)
                base["ann_return_base"] = cagr
        except Exception:
            pass

        # 资产负债率 (debt-to-assets ratio) from 24-h bulk cache
        dr = _fetch_debt_ratio(code)
        if dr is not None:
            base["debt_ratio"] = dr

        return base
    except Exception:
        return {}


@app.route("/api/debug_consensus/<code>")
def debug_consensus(code: str):
    """Debug endpoint: raw Eastmoney consensus API response."""
    try:
        params = {
            "reportName": "RPT_WEB_RESPREDICT",
            "columns":    "WEB_RESPREDICT",
            "filter":     f'(SECURITY_CODE="{code}")',
            "pageSize":   5,
            "source":     "WEB",
            "client":     "WEB",
        }
        r = _em_sess.get(_EM_CONSENSUS_URL, params=params, timeout=8)
        return jsonify({"status": r.status_code, "body": r.json()})
    except Exception as exc:
        return jsonify({"error": str(exc)})


# ── scan state ────────────────────────────────────────────────────────────────
_scan: dict = {"running": False, "done": False, "idx": 0, "total": 0, "results": []}
_scan_lock = threading.Lock()


# ── data helpers ──────────────────────────────────────────────────────────────
def _to_tencent_code(bs_code: str) -> str:
    return bs_code.replace(".", "")


def _fetch_all_codes() -> list[str]:
    cached = _get("all_codes", ttl=86400)
    if cached:
        return cached
    bs.login()
    # Walk back up to 10 days to find the most recent trading day with data
    codes: list[str] = []
    for delta in range(0, 10):
        day = (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")
        rs  = bs.query_all_stock(day=day)
        while rs.next():
            row = rs.get_row_data()
            c = row[0]
            is_sh      = c.startswith("sh.6") and len(c) == 9
            is_sz_main = c.startswith("sz.0") and len(c) == 9
            is_chinext = c.startswith("sz.3") and not c.startswith("sz.39") and len(c) == 9
            if is_sh or is_sz_main or is_chinext:
                codes.append(_to_tencent_code(c))
        if codes:
            break  # found a valid trading day
    bs.logout()
    _set("all_codes", codes)
    return codes


def _fetch_market_caps(codes: list[str], threshold_yi: float) -> list[dict]:
    result = []
    BATCH = 80
    for i in range(0, len(codes), BATCH):
        batch = codes[i : i + BATCH]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        for attempt in range(3):
            try:
                r = _sess.get(url, timeout=10)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        for line in r.text.strip().split(";\n"):
            line = line.strip()
            if not line:
                continue
            try:
                payload = line.split('"')[1]
            except IndexError:
                continue
            fields = payload.split("~")
            if len(fields) < 47:
                continue
            try:
                mktcap = float(fields[44])
            except (ValueError, IndexError):
                continue
            if mktcap < threshold_yi:
                continue
            try:
                price = float(fields[3])
                chg   = float(fields[32]) if len(fields) > 32 else 0.0
            except (ValueError, IndexError):
                price, chg = 0.0, 0.0
            try:
                pe_raw = float(fields[53]) if len(fields) > 53 else 0.0
                # Tencent returns negative PE for loss-making; keep raw value
                pe = round(pe_raw, 1) if -9999 < pe_raw < 9999 else 0.0
            except (ValueError, IndexError):
                pe = 0.0
            code_raw = fields[2]
            market   = "sh" if code_raw.startswith("6") else "sz"
            result.append(
                {
                    "代码":   code_raw,
                    "名称":   fields[1],
                    "市值亿": round(mktcap, 1),
                    "最新价": price,
                    "涨跌幅": round(chg, 2),
                    "pe":     pe,
                    "_tc":    f"{market}{code_raw}",
                }
            )
        time.sleep(0.05)
    result.sort(key=lambda x: x["市值亿"], reverse=True)
    return result


def _fetch_realtime_price(tc_code: str) -> float | None:
    """Return the actual current market price (not 后复权) for a stock."""
    cache_key = f"rt_price_{tc_code}"
    cached = _get(cache_key, ttl=60)   # 1-minute cache
    if cached is not None:
        return cached
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={tc_code}", timeout=5)
        r.raise_for_status()
        for line in r.text.strip().split(";\n"):
            if not line.strip():
                continue
            try:
                fields = line.split('"')[1].split("~")
                price = float(fields[3])
                if price > 0:
                    _set(cache_key, price)
                    return price
            except (IndexError, ValueError):
                pass
    except Exception:
        pass
    return None


def _fetch_kline_hfq(tc_code: str, years: int = 10) -> tuple[list[str], list[float]]:
    today = datetime.now()
    end   = today.strftime("%Y-%m-%d")
    boundaries = []
    t = today - timedelta(days=years * 365 + 10)
    while t < today:
        boundaries.append(t.strftime("%Y-%m-%d"))
        t += timedelta(days=900)
    boundaries.append(end)

    base = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param={code},day,{s},{e},700,hfq"
    )
    dates: list[str] = []
    closes: list[float] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        url = base.format(code=tc_code, s=s, e=e)
        for attempt in range(3):
            try:
                r = _sess.get(url, timeout=10)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        try:
            rows = r.json()["data"][tc_code]["hfqday"]
        except (KeyError, TypeError):
            continue
        for row in rows:
            d = row[0]
            if not dates or d > dates[-1]:
                dates.append(d)
                closes.append(float(row[2]))
    return dates, closes


def _ols(x: np.ndarray, y: np.ndarray):
    A = np.vstack([x, np.ones_like(x)]).T
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yp = s * x + b
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return float(s), float(b), float(r2)


def _fetch_pe_history(
    bs_code: str, years: int = 10
) -> tuple[list[str], list[float], list[str]]:
    """Fetch daily peTTM from BaoStock.
    Returns (pos_dates, pos_pes, loss_dates) where loss_dates are days with
    negative earnings (PE ≤ 0).  bs_code format: 'sh.600900'
    """
    start = (datetime.now() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")
    end   = datetime.now().strftime("%Y-%m-%d")
    bs.login()
    rs = bs.query_history_k_data_plus(
        bs_code, "date,peTTM",
        start_date=start, end_date=end,
        frequency="d", adjustflag="3"
    )
    dates:      list[str]   = []
    pes:        list[float] = []
    loss_dates: list[str]   = []
    while rs.next():
        row = rs.get_row_data()
        raw = (row[1] or "").strip()
        if not raw or raw.lower() in ("none", "nan", ""):
            continue           # no data for this day
        try:
            pe = float(raw)
        except ValueError:
            continue
        if pe < 0:             # negative earnings → loss period
            loss_dates.append(row[0])
        elif 0 < pe < 2000:    # valid positive PE
            dates.append(row[0])
            pes.append(round(pe, 2))
        # pe == 0 or pe >= 2000 → skip extreme outlier
    bs.logout()
    return dates, pes, loss_dates


def _compute_pe_payload(tc_code: str, code: str) -> dict | None:
    """Fetch PE history + decompose price return into EPS-growth vs PE-expansion."""
    cache_key = f"pe_{code}"
    cached = _get(cache_key, ttl=3600)
    if cached:
        return cached

    bs_code = tc_code[:2] + "." + tc_code[2:]   # "sh600900" → "sh.600900"
    pe_dates, pe_vals, loss_dates = _fetch_pe_history(bs_code, years=10)
    if not pe_vals or len(pe_vals) < 50:
        return None

    arr_pe        = np.array(pe_vals, dtype=float)
    pe_current    = float(pe_vals[-1])
    pe_percentile = float(np.mean(arr_pe < pe_current) * 100)

    # Convert consecutive loss dates into contiguous ranges [start, end]
    loss_ranges: list[list[str]] = []
    if loss_dates:
        seg_start = loss_dates[0]
        prev      = loss_dates[0]
        for d in loss_dates[1:]:
            gap = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(prev, "%Y-%m-%d")).days
            if gap > 5:                         # new segment
                loss_ranges.append([seg_start, prev])
                seg_start = d
            prev = d
        loss_ranges.append([seg_start, prev])  # close last segment

    payload: dict = {
        "pe_dates":    pe_dates,
        "pe_vals":     pe_vals,
        "loss_ranges": loss_ranges,             # list of [start_date, end_date]
        "pe_stats": {
            "current":    round(pe_current, 1),
            "median":     round(float(np.median(arr_pe)),          1),
            "p10":        round(float(np.percentile(arr_pe, 10)),  1),
            "p25":        round(float(np.percentile(arr_pe, 25)),  1),
            "p75":        round(float(np.percentile(arr_pe, 75)),  1),
            "p90":        round(float(np.percentile(arr_pe, 90)),  1),
            "percentile": round(pe_percentile, 1),
        },
    }

    # ── Decompose: log(hfq_return) = log(EPS+dividend growth) + log(PE change) ──
    main = _get(f"hist_{code}", ttl=3600)
    if main and main.get("success"):
        price_dates = main["dates"]
        prices_hfq  = main["prices"]
        pe_dict = dict(zip(pe_dates, pe_vals))
        last_ov = next((d for d in reversed(price_dates) if d in pe_dict), None)

        def _build_decomp(start_price_idx: int) -> dict | None:
            """Build decomposition starting from a given price array index."""
            start_pd = price_dates[start_price_idx]
            # nearest PE date on or after start
            first_pe_d = next((d for d in pe_dates if d >= start_pd), None)
            if not first_pe_d or not last_ov or first_pe_d >= last_ov:
                return None
            if first_pe_d not in pe_dict or last_ov not in pe_dict:
                return None
            p_s  = prices_hfq[start_price_idx]
            p_e  = prices_hfq[price_dates.index(last_ov)]
            pe_s = pe_dict[first_pe_d]
            pe_e = pe_dict[last_ov]
            hfq_ret     = p_e / p_s
            pe_ret      = pe_e / pe_s
            eps_div_ret = hfq_ret / pe_ret
            pl = float(np.log(pe_ret))
            el = float(np.log(eps_div_ret))   # = log(hfq_ret) - log(pe_ret)
            # Use |log(EPS)| + |log(PE)| as denominator so that when the two
            # forces cancel out (total ≈ 0) the contributions stay meaningful
            # and always fall in the range [−100%, +100%].
            abs_denom = abs(pl) + abs(el)
            pc = round(pl / abs_denom * 100, 1) if abs_denom > 0.001 else 0.0
            ec = round(el / abs_denom * 100, 1) if abs_denom > 0.001 else 0.0
            return {
                "period_start":        first_pe_d,
                "period_end":          last_ov,
                "pe_start":            round(pe_s, 1),
                "pe_end":              round(pe_e, 1),
                "hfq_total_return":    round((hfq_ret     - 1) * 100, 1),
                "pe_return":           round((pe_ret      - 1) * 100, 1),
                "eps_div_return":      round((eps_div_ret - 1) * 100, 1),
                "pe_contrib_pct":      pc,
                "eps_div_contrib_pct": ec,
            }

        # 10-year: start from first overlapping date
        start_10y = next((i for i, d in enumerate(price_dates) if d in pe_dict), None)
        if start_10y is not None:
            d10 = _build_decomp(start_10y)
            if d10:
                payload["decomposition"] = d10

        # 5-year: start from the regression split index
        split_idx = main.get("split_5y_idx", 0)
        if split_idx and split_idx < len(price_dates):
            d5 = _build_decomp(split_idx)
            if d5:
                payload["decomposition_5y"] = d5

    _set(cache_key, payload)
    return payload


def _compute_regression_payload(tc_code: str, code: str) -> dict | None:
    """Fetch kline + compute both regressions. Returns and caches the full payload."""
    cache_key = f"hist_{code}"
    cached = _get(cache_key, ttl=3600)
    if cached and cached.get("success"):
        return cached

    dates, closes = _fetch_kline_hfq(tc_code, years=10)
    if not dates:
        return None

    # Drop stocks listed less than 3 years ago
    cutoff_3y = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    if dates[0] > cutoff_3y:
        return None

    arr   = np.array(closes, dtype=float)
    log_y = np.log(arr)
    N     = len(log_y)
    x_all = np.arange(N, dtype=float)

    s10, b10, r2_10 = _ols(x_all, log_y)
    reg_10y = np.exp(s10 * x_all + b10).tolist()

    CUT_5Y      = 5 * 365 + 10
    boundary_5y = (datetime.now() - timedelta(days=CUT_5Y)).strftime("%Y-%m-%d")
    split_idx   = next((i for i, d in enumerate(dates) if d >= boundary_5y), N // 2)

    x_5y  = x_all[split_idx:]
    ly_5y = log_y[split_idx:]
    s5, b5, r2_5 = _ols(x_5y, ly_5y)
    reg_5y = np.exp(s5 * x_all + b5).tolist()

    annual_growth_10 = (np.exp(s10 * 252) - 1.0) * 100
    annual_growth_5  = (np.exp(s5  * 252) - 1.0) * 100

    # ── 3-year regression ──────────────────────────────────────────────────────
    CUT_3Y       = 3 * 365 + 10
    boundary_3y  = (datetime.now() - timedelta(days=CUT_3Y)).strftime("%Y-%m-%d")
    split_3y_idx = next((i for i, d in enumerate(dates) if d >= boundary_3y), N - 1)

    x_3y  = x_all[split_3y_idx:]
    ly_3y = log_y[split_3y_idx:]
    s3, b3, r2_3 = _ols(x_3y, ly_3y)
    reg_3y           = np.exp(s3 * x_all + b3).tolist()
    annual_growth_3  = (np.exp(s3 * 252) - 1.0) * 100

    # ── 止盈止损通道（5年 & 3年 残差分布）──────────────────────────────────────
    def _band_stats(s, b, x_w, ly_w, x_all_, arr_, cur_p):
        res   = ly_w - (s * x_w + b)
        p10_  = float(np.percentile(res, 10))
        p90_  = float(np.percentile(res, 90))
        lb    = np.exp(s * x_all_ + b + p10_)
        ub    = np.exp(s * x_all_ + b + p90_)
        dev   = float(np.log(cur_p) - (s * x_all_[-1] + b))
        cw    = p90_ - p10_
        cpos  = float((dev - p10_) / cw * 100) if cw > 0 else 50.0
        pk    = np.maximum.accumulate(arr_)
        mdd   = float(np.min((arr_ - pk) / pk)) * 100
        lbp   = float(lb[-1])
        ubp   = float(ub[-1])
        return lb, ub, {
            "channel_pos":      round(max(0.0, min(200.0, cpos)), 1),
            "max_drawdown":     round(mdd, 1),
            "lower_band_price": round(lbp, 3),
            "upper_band_price": round(ubp, 3),
            "stop_loss_pct":    round((lbp - cur_p) / cur_p * 100, 1),
            "take_profit_pct":  round((ubp - cur_p) / cur_p * 100, 1),
            "lower_pct":        round(p10_ * 100, 1),
            "upper_pct":        round(p90_ * 100, 1),
        }

    cur_price = float(arr[-1])
    lower_band,    upper_band,    band_stats_5 = _band_stats(
        s5, b5, x_5y,  ly_5y,  x_all, arr[split_idx:],    cur_price)
    lower_band_3y, upper_band_3y, band_stats_3 = _band_stats(
        s3, b3, x_3y,  ly_3y,  x_all, arr[split_3y_idx:], cur_price)

    payload = {
        "success":        True,
        "dates":          dates,
        "prices":         [round(v, 4) for v in closes],
        "reg_10y":        [round(v, 4) for v in reg_10y],
        "reg_5y":         [round(v, 4) for v in reg_5y],
        "reg_3y":         [round(v, 4) for v in reg_3y],
        "lower_band":     [round(v, 4) for v in lower_band.tolist()],
        "upper_band":     [round(v, 4) for v in upper_band.tolist()],
        "lower_band_3y":  [round(v, 4) for v in lower_band_3y.tolist()],
        "upper_band_3y":  [round(v, 4) for v in upper_band_3y.tolist()],
        "split_5y_date":  boundary_5y,
        "split_5y_idx":   split_idx,
        "split_3y_date":  boundary_3y,
        "split_3y_idx":   split_3y_idx,
        "stats_10y": {
            "annual_growth": round(annual_growth_10, 2),
            "r_squared":     round(r2_10, 4),
            "total_days":    N,
            "start_price":   float(arr[0]),
            "end_price":     cur_price,
        },
        "stats_5y": {
            "annual_growth": round(annual_growth_5, 2),
            "r_squared":     round(r2_5, 4),
            "total_days":    N - split_idx,
            "start_price":   float(arr[split_idx]),
            "end_price":     cur_price,
        },
        "stats_3y": {
            "annual_growth": round(annual_growth_3, 2),
            "r_squared":     round(r2_3, 4),
            "total_days":    N - split_3y_idx,
            "start_price":   float(arr[split_3y_idx]),
            "end_price":     cur_price,
        },
        "band_stats":    band_stats_5,
        "band_stats_3y": band_stats_3,
    }
    _set(cache_key, payload)
    return payload


# ── scan worker ───────────────────────────────────────────────────────────────
def _run_scan(stock_list: list[dict]):
    with _scan_lock:
        if _scan["running"]:
            return          # already running (shouldn't happen with the new flow)
        # screen_start already set total/results/done; just flip running=True
        _scan["running"] = True

    def process(stock):
        code = stock["代码"]
        tc   = stock.get("_tc", ("sh" if code.startswith("6") else "sz") + code)
        try:
            p = _compute_regression_payload(tc, code)
            if not p:
                return None
            result = {
                "代码":   code,
                "名称":   stock["名称"],
                "市值亿": stock["市值亿"],
                "g10":    p["stats_10y"]["annual_growth"],
                "g5":     p["stats_5y"]["annual_growth"],
                "r10":    round(p["stats_10y"]["r_squared"] * 100, 1),
                "r5":     round(p["stats_5y"]["r_squared"] * 100, 1),
            }
            # Best-effort valuation metrics; use cached price from market-cap list
            vm = _compute_valuation_metrics(code, tc,
                                            stock.get("最新价") or None)
            result.update(vm)   # adds pess_pe / fair_pe_base when available
            return result
        except Exception:
            return None
        finally:
            with _scan_lock:
                _scan["idx"] += 1

    # Pre-warm the 24-h debt-ratio bulk cache before workers start,
    # so that all threads hit the in-memory dict instead of racing to
    # trigger the paginated push2 fetch simultaneously.
    _fetch_all_debt_ratios()

    with ThreadPoolExecutor(max_workers=6) as ex:
        for result in ex.map(process, stock_list):
            if result:
                with _scan_lock:
                    _scan["results"].append(result)

    with _scan_lock:
        _scan["running"] = False
        _scan["done"]    = True


# ── curated ETF list (commodity + sector, liquid A-share ETFs) ───────────────
# is_etf=True skips market-cap / PE filters in the frontend.
# industry_board is used for concentration-penalty grouping.
_MAJOR_ETFS: list[dict] = [
    # 大宗商品
    {"代码": "518880", "名称": "黄金ETF(华安)",   "_tc": "sh518880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159934", "名称": "黄金ETF(易方达)",  "_tc": "sz159934", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159812", "名称": "白银ETF",          "_tc": "sz159812", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159981", "名称": "石油ETF",          "_tc": "sz159981", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159985", "名称": "豆粕ETF",          "_tc": "sz159985", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159980", "名称": "有色金属ETF",      "_tc": "sz159980", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    # 行业 / 主题
    {"代码": "512480", "名称": "半导体ETF",        "_tc": "sh512480", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159995", "名称": "芯片ETF",          "_tc": "sz159995", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515030", "名称": "新能源ETF",        "_tc": "sh515030", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159806", "名称": "新能源车ETF",      "_tc": "sz159806", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512660", "名称": "军工ETF",          "_tc": "sh512660", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515790", "名称": "光伏ETF",          "_tc": "sh515790", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159892", "名称": "储能ETF",          "_tc": "sz159892", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512170", "名称": "医疗ETF",          "_tc": "sh512170", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512880", "名称": "证券ETF",          "_tc": "sh512880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515220", "名称": "煤炭ETF",          "_tc": "sh515220", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159611", "名称": "电力ETF",          "_tc": "sz159611", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512400", "名称": "有色ETF",          "_tc": "sh512400", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "588000", "名称": "科创50ETF",        "_tc": "sh588000", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "159915", "名称": "创业板ETF",        "_tc": "sz159915", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "513130", "名称": "恒生科技ETF",      "_tc": "sh513130", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "513100", "名称": "纳指ETF",          "_tc": "sh513100", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
]


def _fetch_hk_stocks() -> list[dict]:
    """Fetch HK main-board stocks from Eastmoney push2 clist.

    Market cap (f20) is returned in HKD (港元); we convert to 亿港元.
    Liquidity filter: f5 (成交量, 手) ≥ 5000 (≈ 50 万港元/日 for a 10-HKD stock).
    Cached 24 h.
    """
    cache_key = "hk_stocks_v3"   # v3: + f100 industry field
    cached = _get(cache_key, ttl=86400)
    if cached is not None:
        return cached

    result: list[dict] = []
    try:
        PAGE = 500
        page  = 1
        while True:
            r = _em_sess.get(
                # push2delay works where push2 returns connection errors for HK
                "https://push2delay.eastmoney.com/api/qt/clist/get",
                params={
                    "pn":  str(page), "pz": str(PAGE),
                    "po":  "1",       "np": "1",
                    "ut":  "bd1d9ddb04089700cf9c27f6f7426281",
                    "fltt": "2",      "invt": "2", "fid": "f20",
                    "fs":  "m:128+t:3,m:128+t:4",
                    "fields": "f12,f14,f20,f9,f5,f100,f3",
                },
                timeout=15,
            )
            data = r.json().get("data") or {}
            diff = data.get("diff") or []
            if not diff:
                break

            for item in diff:
                code_raw = str(item.get("f12") or "").strip()
                if not code_raw:
                    continue
                code = code_raw.zfill(5)

                # f20 = 总市值 in 港元 (raw HKD); ÷1e8 → 亿港元
                # Confirmed: 腾讯 f20≈3.9e12 → 38952亿港元
                try:
                    mktcap_hkd = float(item.get("f20") or 0) / 1e8
                except (TypeError, ValueError):
                    continue
                if mktcap_hkd <= 0:
                    continue

                # Liquidity: f5 = shares traded per day; require ≥100 000 shares
                try:
                    if float(item.get("f5") or 0) < 100_000:
                        continue
                except (TypeError, ValueError):
                    continue

                try:
                    pe = round(float(item.get("f9") or 0), 1)
                    if abs(pe) > 9999:
                        pe = 0.0
                except (TypeError, ValueError):
                    pe = 0.0

                industry_hk = str(item.get("f100") or "").strip()
                try:
                    chg_hk = round(float(item.get("f3") or 0), 2)
                except (TypeError, ValueError):
                    chg_hk = 0.0
                result.append({
                    "代码":   code,
                    "名称":   str(item.get("f14") or ""),
                    "市值亿": round(mktcap_hkd, 1),
                    "pe":     pe,
                    "_tc":    f"hk{code}",
                    "is_hk":  True,
                    "最新价":  0,
                    "涨跌幅": chg_hk,
                    "industry_board": industry_hk,
                })

            total = data.get("total") or 0
            if page * PAGE >= total:
                break
            page += 1
            time.sleep(0.3)
    except Exception:
        pass

    if result:
        _set(cache_key, result)
    return result


# ── trend trading scan ────────────────────────────────────────────────────────

def _fetch_recent_highs(tc_code: str) -> tuple[float, float, float, float]:
    """Return (max_close_50d, max_close_50d, intraday_price, atr_50d).

    intraday_price = yesterday's hfq close scaled by today's real-time chg%,
    so it stays on the same hfq basis as max_close_50d.
    Kline data (max_c50, last_hfq, atr) cached 2 h; real-time price fetched fresh.
    Returns (0.0, 0.0, 0.0, 0.0) on failure.
    """
    # ── slow-changing kline data: cached 2 h ─────────────────────────────────
    cache_key = f"rec_highs_v4_{tc_code}"
    cached = _get(cache_key, ttl=7200)
    if cached is not None:
        max_c50, last_hfq, atr = cached
    else:
        max_c50, last_hfq, atr = 0.0, 0.0, 0.0
        try:
            today = datetime.now()
            start = (today - timedelta(days=80)).strftime("%Y-%m-%d")  # ~55 trading days
            end   = today.strftime("%Y-%m-%d")
            url   = (
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={tc_code},day,{start},{end},75,hfq"
            )
            r = _sess.get(url, timeout=8)
            r.raise_for_status()
            sd     = r.json()["data"][tc_code]
            rows   = (sd.get("hfqday") or sd.get("qfqday") or sd.get("day") or [])
            closes = [float(row[2]) for row in rows if len(row) > 2 and row[2]]
            if len(closes) >= 10:
                max_c50  = max(closes[-50:]) if len(closes) >= 50 else max(closes)
                last_hfq = closes[-1]
                rets = [abs(closes[i] / closes[i-1] - 1) * 100
                        for i in range(1, len(closes))]
                atr  = round(sum(rets[-50:]) / min(len(rets), 50), 2)
                _set(cache_key, [round(max_c50, 4), round(last_hfq, 4), atr])
        except Exception:
            pass

    if last_hfq == 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    # ── real-time intraday price: apply today's chg% to hfq last close ───────
    cur_price = last_hfq
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={tc_code}", timeout=5)
        for ln in r.text.strip().split(";\n"):
            if not ln.strip():
                continue
            flds = ln.split('"')[1].split("~")
            if len(flds) > 32 and flds[32]:
                chg = float(flds[32])          # today's % change (e.g. -6.20)
                cur_price = round(last_hfq * (1 + chg / 100), 4)
            break
    except Exception:
        pass

    return (round(max_c50, 4), round(max_c50, 4), cur_price, atr)


def _weekly_end_date() -> str:
    """Return the end date to use for weekly K-line queries.

    A weekly bar is considered complete once Friday's A-share session has
    closed (15:00 CST).  Rules:
      - Sat / Sun              → this past Friday is complete
      - Fri after 15:00        → today (Friday) is complete
      - Mon–Thu, or Fri <15:00 → last week's Friday (current week incomplete)
    """
    now = datetime.now()
    wd  = now.weekday()          # 0 = Mon … 6 = Sun
    if wd == 5:                  # Saturday
        last_fri = now - timedelta(days=1)
    elif wd == 6:                # Sunday
        last_fri = now - timedelta(days=2)
    elif wd == 4 and now.hour >= 16:   # Friday after close (HK closes 16:00, A-share 15:00)
        last_fri = now
    else:
        # Mon–Thu: go back to last Friday
        # Fri before 15:00: go back 7 days to previous Friday
        days_back = (wd - 4) % 7 or 7
        last_fri = now - timedelta(days=days_back)
    return last_fri.strftime("%Y-%m-%d")


def _fetch_hk_weekly_closes_em(hk_code: str, n: int, end: str) -> list[float]:
    """Fetch last n HK weekly hfq-adjusted closes from Eastmoney push2his.

    hk_code: Tencent-style code like 'hk02380'.
    end:     YYYY-MM-DD cutoff (last completed Friday from _weekly_end_date()).
    Returns: list of float (closing prices), oldest first, length <= n.
    """
    raw_code = hk_code[2:].lstrip("0") or "0"   # 'hk02380' → '2380'
    secid    = f"128.{hk_code[2:]}"              # Eastmoney secid: 128.02380
    beg_date = (datetime.strptime(end, "%Y-%m-%d")
                - timedelta(days=(n + 6) * 7)).strftime("%Y%m%d")
    end_date = end.replace("-", "")
    try:
        r = _sess.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid":   secid,
                "fields1": "f1,f2,f3,f4,f5",
                "fields2": "f51,f52,f53,f54,f55,f56",
                "klt":     "102",    # weekly
                "fqt":     "1",      # 后复权
                "beg":     beg_date,
                "end":     end_date,
                "lmt":     str(n + 10),
            },
            headers={"Referer": "https://quote.eastmoney.com/"},
            timeout=10,
        )
        r.raise_for_status()
        klines = ((r.json().get("data") or {}).get("klines") or [])
        closes = []
        for k in klines:
            parts = k.split(",")
            if len(parts) < 3:
                continue
            date_str = parts[0]
            if date_str.replace("-", "") > end_date:   # drop incomplete week
                continue
            try:
                closes.append(float(parts[2]))         # field: close
            except (ValueError, IndexError):
                pass
        return closes[-n:]
    except Exception:
        return []


def _fetch_weekly_closes_long(tc_code: str, n: int = 130) -> list[float]:
    """Return last n weekly hfq-adjusted closing prices for percentile calculations.

    Uses a separate cache key from _fetch_weekly_closes so that the short
    22-week cache is not invalidated.  Cached for 2 h.
    """
    cache_key = f"weekly_long_v1_{tc_code}"
    cached = _get(cache_key, ttl=7200)
    if cached is not None:
        return cached
    result: list[float] = []
    try:
        end   = _weekly_end_date()
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=(n + 6) * 7)).strftime("%Y-%m-%d")
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tc_code},week,{start},{end},{n + 6},hfq"
        )
        r = _sess.get(url, timeout=10)
        r.raise_for_status()
        sd   = r.json()["data"][tc_code]
        rows = (sd.get("hfqweek") or sd.get("qfqweek") or sd.get("week") or [])
        if rows and rows[-1][0] > end:
            rows = rows[:-1]
        closes = [float(row[2]) for row in rows if row[2]]
        result = closes[-n:]
    except Exception:
        pass
    if result:
        _set(cache_key, result)
    return result


def _fetch_weekly_closes(tc_code: str, n: int = 12) -> list[float]:
    """Return last n weekly hfq-adjusted closing prices, oldest first.

    Uses Tencent's fqkline 'week' endpoint.  Cached for 2 h.
    """
    cache_key = f"weekly_v2_{tc_code}"
    cached = _get(cache_key, ttl=7200)
    if cached is not None:
        return cached
    result: list[float] = []
    try:
        end   = _weekly_end_date()
        # Request enough history to guarantee n complete weeks
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=(n + 6) * 7)).strftime("%Y-%m-%d")
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tc_code},week,{start},{end},{n + 6},hfq"
        )
        r = _sess.get(url, timeout=8)
        r.raise_for_status()
        sd   = r.json()["data"][tc_code]
        rows = (sd.get("hfqweek") or sd.get("qfqweek") or sd.get("week") or [])
        if rows and rows[-1][0] > end:
            rows = rows[:-1]
        closes = [float(row[2]) for row in rows if row[2]]
        result = closes[-n:]
    except Exception:
        pass
    if result:
        _set(cache_key, result)
    return result


# Trend scan state (independent of the main screener scan)
_trend_scan: dict = {
    "running": False, "done": False, "idx": 0, "total": 0, "results": []
}
_trend_lock = threading.Lock()


def _run_trend_scan(stock_list: list[dict]) -> None:
    with _trend_lock:
        if _trend_scan["running"]:
            return
        _trend_scan["running"] = True

    def process(stock):
        code   = stock["代码"]
        tc     = stock.get("_tc", ("sh" if code.startswith("6") else "sz") + code)
        is_etf = bool(stock.get("is_etf"))
        is_hk  = bool(stock.get("is_hk"))
        try:
            n_weeks  = 7 if is_etf else 10   # ETF: shorter window for fairer comparison
            closes_all = _fetch_weekly_closes(tc, n=22)   # always fetch 22 for ret20
            if len(closes_all) < 5:
                return None
            closes = closes_all[-n_weeks:]    # regression window
            n_pts  = len(closes)
            x = np.arange(n_pts, dtype=float)
            y = np.log(np.array(closes, dtype=float))
            slope, b_intercept, r2 = _ols(x, y)

            # σ_res: std dev of log-price residuals — slope-independent trend quality
            y_pred     = slope * x + b_intercept
            sigma_res  = round(float(np.std(y - y_pred)) * 100, 3)   # in %

            # Normalised closes for sparkline (divide by first close → starts at 1.0)
            p0 = closes[0]
            norm = [round(p / p0, 4) for p in closes]

            # Industry: ETFs use pre-set category; HK stocks try PUBLISHNAME; A-shares normal
            if is_etf:
                industry = stock.get("industry_board", "ETF")
            elif is_hk:
                industry = stock.get("industry_board", "") or "港股其他"
            else:
                industry = _fetch_industry_board(code)

            h7, h30, last_cls, atr_50d = _fetch_recent_highs(tc)

            # 20-week return (for overextension penalty in UI)
            ret20w = None
            if len(closes_all) >= 20:
                c0, c1 = closes_all[-20], closes_all[-1]
                if c0 > 0:
                    ret20w = round((c1 / c0 - 1) * 100, 1)

            # pct10w / pct20w / closes_52w are fetched on-demand via
            # /api/trend/closes52w after the frontend filters results down
            # to a small set — no need to fetch 130 weeks for all 2000+ stocks.

            # Today's change % — passed in from stock list for A/HK; fetch for ETFs
            chg_today = stock.get("涨跌幅")
            if chg_today is None:
                try:
                    r = _sess.get(f"https://qt.gtimg.cn/q={tc}", timeout=5)
                    for ln in r.text.strip().split(";\n"):
                        if not ln.strip():
                            continue
                        flds = ln.split('"')[1].split("~")
                        if len(flds) > 32:
                            chg_today = round(float(flds[32]), 2)
                        break
                except Exception:
                    pass

            return {
                "代码":          code,
                "名称":          stock["名称"],
                "市值亿":        stock["市值亿"],
                "pe":            stock.get("pe", 0.0),
                "is_etf":        is_etf,
                "is_hk":         is_hk,
                "_tc":           tc,
                "trend_slope":   round(slope * 100, 2),
                "trend_r2":      round(r2 * 100, 1),
                "sigma_res":     sigma_res,
                "trend_closes":  norm,
                "industry_board": industry,
                "max_high_7d":   h7,
                "max_high_30d":  h30,
                "last_close":    last_cls,
                "atr_50d":       atr_50d,
                "ret20w":        ret20w,
                "涨跌幅":        chg_today,
            }
        except Exception:
            return None
        finally:
            with _trend_lock:
                _trend_scan["idx"] += 1

    # 4 workers: polite to Tencent's fqkline API; avoids rate-limit 429s
    with ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(process, stock_list):
            if result:
                with _trend_lock:
                    _trend_scan["results"].append(result)

    with _trend_lock:
        _trend_scan["running"] = False
        _trend_scan["done"]    = True


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    return render_template("mobile.html")


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "趋势选股",
        "short_name": "选股",
        "start_url": "/mobile",
        "display": "standalone",
        "background_color": "#0d1117",
        "theme_color": "#0d1117",
        "icons": [
            {"src": "https://fav.farm/%F0%9F%93%88", "sizes": "192x192", "type": "image/png"},
            {"src": "https://fav.farm/%F0%9F%93%88", "sizes": "512x512", "type": "image/png"},
        ]
    })


@app.route("/api/stocks")
def get_stocks():
    cached = _get("stock_list", ttl=86400)
    if cached:
        return jsonify(cached)
    try:
        codes  = _fetch_all_codes()
        stocks = _fetch_market_caps(codes, threshold_yi=300.0)
        payload = {"success": True, "data": stocks, "total": len(stocks)}
        _set("stock_list", payload)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stock/<code>")
def get_stock_data(code: str):
    try:
        tc_code = ("sh" if code.startswith("6") else "sz") + code
        cl = _get("stock_list", ttl=86400)
        if cl:
            for s in cl["data"]:
                if s["代码"] == code:
                    tc_code = s["_tc"]
                    break
        p = _compute_regression_payload(tc_code, code)
        if not p:
            return jsonify({"success": False, "error": "无历史数据"}), 404
        # Merge PE payload (BaoStock peTTM + decomposition)
        pe = _compute_pe_payload(tc_code, code)
        if pe:
            p = {**p, **pe}
        # Analyst consensus EPS forecasts (Eastmoney, best-effort)
        consensus = _fetch_consensus_eps(code)
        # Compute forward PE using actual (non-后复权) market price
        actual_price = _fetch_realtime_price(tc_code)
        if actual_price and actual_price > 0:
            for fc in consensus.get("forecasts", []):
                if fc["eps"] and fc["eps"] > 0:
                    fc["forward_pe"] = round(actual_price / fc["eps"], 1)
            # ── Pessimistic blended Forward PE ─────────────────────────────
            # A = most recent actual EPS; B/C/D = consensus estimates
            # B* = √(A·B)  C* = ∛(A·B·C)  D* = ∜(A·B·C·D)
            # Pessimistic FwdPE = price × 4 / (A + B* + C* + D*)
            fcs = sorted(consensus.get("forecasts", []), key=lambda x: x["year"])
            actual_fcs  = [f for f in fcs if f.get("year_mark") == "A"]
            est_fcs     = [f for f in fcs if f.get("year_mark") == "E"]
            if actual_fcs and len(est_fcs) >= 3:
                A = actual_fcs[-1]["eps"]
                B = est_fcs[0]["eps"]
                C = est_fcs[1]["eps"]
                D = est_fcs[2]["eps"]
                if all(v > 0 for v in [A, B, C, D]):
                    B_star = (A * B) ** 0.5
                    C_star = (A * B * C) ** (1 / 3)
                    D_star = (A * B * C * D) ** 0.25
                    pess_pe = actual_price * 4 / (A + B_star + C_star + D_star)
                    consensus["pessimistic_fwd_pe"] = round(pess_pe, 1)
                    # 3Y CAGR: pessimistic (D*/A) and consensus (D/A)
                    pess_g = ((D_star / A) ** (1 / 3) - 1) * 100
                    cons_g = ((D     / A) ** (1 / 3) - 1) * 100
                    consensus["pessimistic_g"] = round(pess_g, 2)
                    consensus["consensus_g"]   = round(cons_g, 2)
                    # ── Fair PE via Gordon Growth Model ─────────────────────
                    # ── Two-stage DCF Fair PE ───────────────────────────────
                    # Stage 1 (i=1,2,3): discount pessimistic EPS (B*, C*, D*)
                    # Stage 2 (terminal): perpetual growth at g_t after year 3
                    # PE_fair = Σ (E_i/E0)/(1+r)^i  +  E3/E0·(1+g_t)/[(r-g_t)·(1+r)^3]
                    rf  = _fetch_cn10y_yield()
                    erp = 5.0   # equity risk premium for A-shares
                    r   = rf + erp    # in %

                    r_dec   = r / 100
                    ratios  = [B_star / A, C_star / A, D_star / A]
                    pv_s1   = sum(er / (1 + r_dec) ** i
                                  for i, er in enumerate(ratios, start=1))
                    disc3   = (1 + r_dec) ** 3

                    # Dividend yield (best-effort; 0 if unavailable)
                    div_yield     = _fetch_dividend_yield(code) or 0.0
                    # Cap at 8 % to prevent one-off special dividends (e.g. 茅台
                    # 2024 特别分红) from inflating multi-year return projections.
                    # The raw TTM yield is preserved in fair_pe_analysis for display.
                    DIV_YIELD_CAP = 8.0
                    div_yield_calc = min(div_yield, DIV_YIELD_CAP)
                    annual_div     = div_yield_calc / 100 * actual_price  # ¥ per year

                    def _two_stage(g_t_pct):
                        g_t = g_t_pct / 100
                        if g_t >= r_dec:
                            return None, None
                        tv        = ratios[-1] * (1 + g_t) / (r_dec - g_t) / disc3
                        fair_pe   = round(pv_s1 + tv, 1)
                        # Terminal price at end of Year-3: the Gordon-growth
                        # perpetuity value of D★ growing at g_t forever
                        target_p3 = D_star * (1 + g_t) / (r_dec - g_t)
                        return fair_pe, round(target_p3, 2)

                    def _ann_return(target_p3):
                        """3-year pure capital gain: (P3/P0)^(1/3) − 1.

                        Under Modigliani-Miller, the Gordon-Growth terminal price P3 already
                        capitalises the full earnings power of the firm (D★ in the numerator
                        treats EPS as if fully paid out or reinvested equivalently).  Adding
                        actual dividends on top would double-count the payout portion.
                        Dividend yield is displayed separately in the UI.
                        """
                        if not target_p3 or actual_price <= 0:
                            return None
                        return round(((target_p3 / actual_price) ** (1 / 3) - 1) * 100, 1)

                    # Implied terminal g: solve analytically (fixed r, solve g_t)
                    # pess_pe = pv_s1 + ratios[-1]·(1+g_t)/[(r-g_t)·disc3]
                    # let R = (pess_pe - pv_s1)·disc3 / ratios[-1]
                    # => g_t = (R·r_dec - 1) / (R + 1)
                    implied_terminal_g = None
                    try:
                        R = (pess_pe - pv_s1) * disc3 / ratios[-1]
                        if R > 0:
                            implied_terminal_g = round(
                                (R * r_dec - 1) / (R + 1) * 100, 2)
                    except Exception:
                        pass

                    # Implied discount rate: fix g_t = 0 %, solve for r via bisection
                    # Answers: "what total required return does the market demand for
                    # this stock, assuming neutral (0 %) long-run growth?"
                    implied_r = None
                    try:
                        def _fair_pe_at_r(r_try):
                            d3 = (1 + r_try) ** 3
                            s1 = sum(ratios[i] / (1 + r_try) ** (i + 1)
                                     for i in range(3))
                            # g_t = 0 simplifies tv = ratios[-1] / (r * d3)
                            return s1 + ratios[-1] / (r_try * d3)

                        lo, hi = 0.01, 0.80        # search r in [1 %, 80 %]
                        target = pess_pe
                        if _fair_pe_at_r(lo) >= target >= _fair_pe_at_r(hi):
                            for _ in range(60):    # ~60 iterations → precision < 1e-6
                                mid = (lo + hi) / 2
                                if _fair_pe_at_r(mid) > target:
                                    lo = mid
                                else:
                                    hi = mid
                            implied_r = round((lo + hi) / 2 * 100, 2)   # in %
                    except Exception:
                        pass

                    fp0, tp0 = _two_stage(0.0)
                    fp1, tp1 = _two_stage(rf)
                    fp2, tp2 = _two_stage(3.0)

                    # "No-rerating" baseline: what if the stock stays at its
                    # current trailing PE in 3 years?  P3_hold = trailing_PE × D★.
                    # Isolates EPS-growth return from PE-rerating return.
                    # (Same MM-consistent pure capital gain — dividends shown separately.)
                    trailing_pe     = actual_price / A
                    target_p3_hold  = trailing_pe * D_star     # constant-PE target
                    ann_return_hold = round(
                        ((target_p3_hold / actual_price) ** (1 / 3) - 1) * 100, 1
                    ) if actual_price > 0 else 0.0

                    consensus["fair_pe_analysis"] = {
                        "rf":               rf,
                        "erp":              erp,
                        "r":                round(r, 2),
                        "pess_pe":          round(pess_pe, 1),
                        "base_eps":         round(A, 2),
                        "d_star":           round(D_star, 2),
                        "actual_price":     round(actual_price, 2),
                        "trailing_pe":      round(trailing_pe, 1),
                        "ann_return_hold":  ann_return_hold,
                        "implied_r":        implied_r,         # market-implied discount rate (g_t=0%)
                        "div_yield":        round(div_yield, 2),
                        "div_yield_calc":   round(div_yield_calc, 2),
                        "div_yield_capped": div_yield > DIV_YIELD_CAP,
                        "implied_terminal_g": implied_terminal_g,
                        "scenarios": [
                            {"label": "悲观终局",
                             "g_t":  0.0,
                             "fair_pe": fp0,
                             "target_price_3yr": tp0,
                             "annualized_return": _ann_return(tp0)},
                            {"label": "基准终局",
                             "g_t":  round(rf, 1),
                             "fair_pe": fp1,
                             "target_price_3yr": tp1,
                             "annualized_return": _ann_return(tp1)},
                            {"label": "温和终局",
                             "g_t":  3.0,
                             "fair_pe": fp2,
                             "target_price_3yr": tp2,
                             "annualized_return": _ann_return(tp2)},
                        ],
                    }
                    # Store adjusted EPS values for display
                    consensus["pessimistic_eps"] = {
                        "A":      round(A, 2),
                        "B_star": round(B_star, 2),
                        "C_star": round(C_star, 2),
                        "D_star": round(D_star, 2),
                        "years":  [actual_fcs[-1]["year"],
                                   est_fcs[0]["year"],
                                   est_fcs[1]["year"],
                                   est_fcs[2]["year"]],
                    }
        # ── P/B analysis for financial stocks ──────────────────────────────
        industry = consensus.get("industry_board", "")
        if _is_financial(industry) and actual_price and actual_price > 0:
            pb = _fetch_realtime_pb(tc_code)
            if pb and pb > 0:
                bv     = actual_price / pb
                rf_fin = _fetch_cn10y_yield()
                erp_fin = _FINANCIAL_ERP
                r_fin  = rf_fin + erp_fin
                r_fin_dec = r_fin / 100

                # ROE from most recent actual EPS / current BV
                actual_eps = next(
                    (f["eps"] for f in consensus.get("forecasts", [])
                     if f.get("year_mark") == "A"),
                    None
                )
                roe = round(actual_eps / bv * 100, 1) if actual_eps and bv > 0 else None

                def _fair_pb(g_t_pct: float) -> float | None:
                    g_t = g_t_pct / 100
                    if g_t >= r_fin_dec or roe is None:
                        return None
                    return round((roe / 100) / (r_fin_dec - g_t), 2)

                # Implied required return: r = ROE/PB + g  (base: g = rf)
                implied_r = round(
                    (roe / 100) / pb * 100 + rf_fin, 1
                ) if roe else None

                consensus["pb_analysis"] = {
                    "current_pb":  round(pb, 2),
                    "bv_per_share": round(bv, 1),
                    "roe":         roe,
                    "rf":          rf_fin,
                    "erp":         erp_fin,
                    "r":           round(r_fin, 2),
                    "implied_r":   implied_r,
                    "industry":    industry,
                    "scenarios": [
                        {"label": "悲观终局", "g_t": 0.0,
                         "fair_pb": _fair_pb(0.0)},
                        {"label": "基准终局", "g_t": round(rf_fin, 1),
                         "fair_pb": _fair_pb(rf_fin)},
                        {"label": "温和终局", "g_t": 3.0,
                         "fair_pb": _fair_pb(3.0)},
                    ],
                }

        # ── Cyclical stock: historical mean-PE valuation ───────────────────────
        # For coal / steel / non-ferrous metals etc., DCF can give misleading
        # results when current EPS is at a cyclical peak or trough.
        # Use historical P10/P25/median/P75 PE × current actual EPS instead.
        if _is_cyclical(industry) and actual_price and actual_price > 0 and pe:
            pe_stats   = pe.get("pe_stats", {})
            pe_median  = pe_stats.get("median")
            pe_p25     = pe_stats.get("p25")
            pe_p75     = pe_stats.get("p75")
            pe_current = pe_stats.get("current")
            pe_pct     = pe_stats.get("percentile")

            # Most recent actual EPS
            actual_eps_cyc = next(
                (f["eps"] for f in consensus.get("forecasts", [])
                 if f.get("year_mark") == "A"),
                None,
            )

            if pe_median and actual_eps_cyc and actual_eps_cyc > 0:
                # "Implied normalised EPS": what per-share earnings the market
                # is pricing in at the historical median multiple.
                # If current EPS >> implied → cycle peak; << implied → cycle trough.
                implied_norm_eps = round(actual_price / pe_median, 2)
                eps_premium_pct  = round(
                    (actual_eps_cyc / implied_norm_eps - 1) * 100, 1
                ) if implied_norm_eps > 0 else None

                def _fp(mult):
                    if mult is None:
                        return None
                    return round(actual_eps_cyc * mult, 2)

                def _vsp(fp):
                    if fp is None or actual_price <= 0:
                        return None
                    return round((fp / actual_price - 1) * 100, 1)

                fp25 = _fp(pe_p25);  fp50 = _fp(pe_median);  fp75 = _fp(pe_p75)
                consensus["cyclical_analysis"] = {
                    "industry":          industry,
                    "current_price":     round(actual_price, 2),
                    "current_eps":       round(actual_eps_cyc, 2),
                    "current_pe":        pe_current,
                    "pe_percentile":     pe_pct,
                    "pe_p25":            pe_p25,
                    "pe_median":         pe_median,
                    "pe_p75":            pe_p75,
                    "implied_norm_eps":  implied_norm_eps,
                    "eps_premium_pct":   eps_premium_pct,
                    "scenarios": [
                        {"label": "低谷 (P25)",  "pe": pe_p25,   "fair_price": fp25, "vs_price": _vsp(fp25)},
                        {"label": "均值 (中位)", "pe": pe_median, "fair_price": fp50, "vs_price": _vsp(fp50)},
                        {"label": "景气 (P75)",  "pe": pe_p75,   "fair_price": fp75, "vs_price": _vsp(fp75)},
                    ],
                }

        p["consensus_eps"] = consensus
        return jsonify(p)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


def _resolve_tc(code: str) -> str:
    tc = ("sh" if code.startswith("6") else "sz") + code
    cl = _get("stock_list", ttl=86400)
    if cl:
        for s in cl["data"]:
            if s["代码"] == code:
                return s["_tc"]
    return tc


@app.route("/api/stock/kline/<code>")
def get_stock_kline(code: str):
    """Return hfq OHLC for daily (6 months) and weekly (2 years)."""
    tc = _resolve_tc(code)
    today = datetime.now()
    end   = today.strftime("%Y-%m-%d")

    def _fetch(freq, start, count, key):
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tc},{freq},{start},{end},{count},hfq"
        )
        try:
            r  = _sess.get(url, timeout=10)
            sd = r.json()["data"][tc]
            rows = sd.get(f"hfq{freq}") or sd.get(f"qfq{freq}") or sd.get(freq) or []
            out = []
            for row in rows:
                if len(row) >= 5:
                    out.append({
                        "date":   row[0],
                        "open":   float(row[1]),
                        "close":  float(row[2]),
                        "high":   float(row[3]),
                        "low":    float(row[4]),
                        "volume": float(row[5]) if len(row) > 5 else 0,
                    })
            return out
        except Exception:
            return []

    start_d = (today - timedelta(days=200)).strftime("%Y-%m-%d")
    start_w = (today - timedelta(days=730)).strftime("%Y-%m-%d")
    daily   = _fetch("day",  start_d, 150, "hfqday")
    weekly  = _fetch("week", start_w, 104, "hfqweek")
    return jsonify({"success": True, "daily": daily, "weekly": weekly})


@app.route("/api/stock/intraday/<code>")
def get_stock_intraday(code: str):
    """Return today's minute price series."""
    tc = _resolve_tc(code)
    try:
        url  = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={tc}"
        r    = _sess.get(url, timeout=10)
        blob = r.json()
        d    = blob.get("data", {}).get(tc, {})
        inner = d.get("data", {})
        raw   = inner.get("data", [])
        # prev_close: try inner.preClose, fallback to qt array index 4
        prev_close = 0.0
        try:
            qt_arr = d.get("qt", {}).get(tc, [])
            if len(qt_arr) > 4 and qt_arr[4]:
                prev_close = float(qt_arr[4])
        except Exception:
            pass
        if not prev_close:
            try:
                prev_close = float(inner.get("preClose") or 0)
            except Exception:
                pass
        times, prices = [], []
        for item in raw:
            if isinstance(item, str):
                parts = item.split()
                if len(parts) >= 2:
                    t = parts[0].zfill(4)
                    times.append(f"{t[:2]}:{t[2:]}")
                    prices.append(float(parts[1]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                t = str(item[0]).zfill(4)
                times.append(f"{t[:2]}:{t[2:]}")
                prices.append(float(item[1]))
        return jsonify({"success": True, "times": times, "prices": prices,
                        "prev_close": prev_close})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/screen/start", methods=["POST"])
def screen_start():
    cl = _get("stock_list", ttl=86400)
    if not cl:
        return jsonify({"error": "请先加载股票列表"}), 400
    force = (request.args.get("force") == "1") or (
        (request.get_json(silent=True) or {}).get("force", False)
    )
    with _scan_lock:
        already_running = _scan["running"]
        already_done    = _scan["done"] and len(_scan["results"]) > 0

    if already_done and not force:
        return jsonify({"ok": True, "cached": True, "total": _scan["total"]})

    if not already_running:
        stock_list = cl["data"]
        # Atomically reset ALL state (including total) before starting the
        # thread so that poll requests during the startup window never see
        # total=0 and never trigger the "done" branch prematurely.
        with _scan_lock:
            _scan.update({
                "running": False, "done": False, "idx": 0,
                "total":   len(stock_list), "results": [],
            })
        threading.Thread(target=_run_scan, args=(stock_list,), daemon=True).start()
    return jsonify({"ok": True, "cached": False, "total": len(cl["data"])})


@app.route("/api/screen/progress")
def screen_progress():
    with _scan_lock:
        return jsonify({
            "running": _scan["running"],
            "done":    _scan["done"],
            "idx":     _scan["idx"],
            "total":   _scan["total"],
            "results": list(_scan["results"]),
        })


def _build_trend_stock_list() -> dict:
    """Build (or return cached) the trend-scan universe.

    A-shares ≥100亿  +  curated ETFs  +  HK main-board 100-2000亿HKD.
    Completely independent of the main stock_list (300亿 A-share list).
    Cached 24 h under 'trend_stock_list_v3'.
    """
    cached = _get("trend_stock_list_v3", ttl=86400)
    if cached:
        return cached

    codes    = _fetch_all_codes()
    a_stocks = _fetch_market_caps(codes, threshold_yi=100.0)
    a_codes  = {s["代码"] for s in a_stocks}
    etfs     = [e for e in _MAJOR_ETFS if e["代码"] not in a_codes]
    hk       = _fetch_hk_stocks()

    payload = {
        "success": True,
        "data":    a_stocks + etfs + hk,
        "total":   len(a_stocks) + len(etfs) + len(hk),
    }
    _set("trend_stock_list_v3", payload)
    return payload


@app.route("/api/trend/stocks")
def trend_stocks():
    """HTTP endpoint that returns (or builds) the trend-scan universe."""
    try:
        return jsonify(_build_trend_stock_list())
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/trend/start", methods=["POST"])
def trend_start():
    # Build the list if needed — never falls back to the main 300亿 stock_list
    try:
        cl = _build_trend_stock_list()
    except Exception as exc:
        return jsonify({"error": f"加载股票列表失败：{exc}"}), 400
    if not cl.get("data"):
        return jsonify({"error": "股票列表为空，请稍后重试"}), 400
    force = (request.args.get("force") == "1") or (
        (request.get_json(silent=True) or {}).get("force", False)
    )
    with _trend_lock:
        already_running = _trend_scan["running"]
        already_done    = _trend_scan["done"] and len(_trend_scan["results"]) > 0
    if already_done and not force:
        return jsonify({"ok": True, "cached": True, "total": _trend_scan["total"]})
    if not already_running:
        stock_list = cl["data"]
        with _trend_lock:
            _trend_scan.update({
                "running": False, "done": False, "idx": 0,
                "total":   len(stock_list), "results": [],
            })
        threading.Thread(
            target=_run_trend_scan, args=(stock_list,), daemon=True
        ).start()
    return jsonify({"ok": True, "cached": False, "total": len(cl["data"])})


@app.route("/api/trend/closes52w", methods=["POST"])
def trend_closes52w():
    """Fetch 52-week closes + rolling percentiles for a small filtered batch.

    Request:  {"stocks": [{"代码": "600000", "_tc": "sh600000"}, ...]}
    Response: {"600000": {"closes_52w": [...52 floats...],
                          "pct10w": 75.0, "pct20w": 60.0}, ...}
    """
    body   = request.get_json(silent=True) or {}
    stocks = body.get("stocks", [])
    if not stocks:
        return jsonify({})

    def _compute_one(s):
        code = s.get("代码", "")
        tc   = s.get("_tc", ("sh" if code.startswith("6") else "sz") + code)
        try:
            lc  = _fetch_weekly_closes_long(tc, n=130)
            arr = np.array(lc, dtype=float)
            n   = len(arr)

            closes_52w = None
            if n >= 52 and arr[-52] > 0:
                closes_52w = [round(p / arr[-52], 4) for p in arr[-52:]]

            pct10w = None
            if n >= 11:
                rets10 = [(arr[i+10]/arr[i] - 1)*100
                          for i in range(n-10) if arr[i] > 0]
                cur10  = (arr[-1]/arr[-11] - 1)*100 if arr[-11] > 0 else None
                if rets10 and cur10 is not None:
                    pct10w = round(float(np.mean(np.array(rets10) <= cur10))*100, 1)

            pct20w = None
            if n >= 21:
                rets20 = [(arr[i+20]/arr[i] - 1)*100
                          for i in range(n-20) if arr[i] > 0]
                cur20  = (arr[-1]/arr[-21] - 1)*100 if arr[-21] > 0 else None
                if rets20 and cur20 is not None:
                    pct20w = round(float(np.mean(np.array(rets20) <= cur20))*100, 1)

            return code, {"closes_52w": closes_52w, "pct10w": pct10w, "pct20w": pct20w}
        except Exception:
            return code, {}

    result = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for code, data in ex.map(_compute_one, stocks):
            if code:
                result[code] = data
    return jsonify(result)


@app.route("/api/trend/progress")
def trend_progress():
    with _trend_lock:
        return jsonify({
            "running": _trend_scan["running"],
            "done":    _trend_scan["done"],
            "idx":     _trend_scan["idx"],
            "total":   _trend_scan["total"],
            "results": list(_trend_scan["results"]),
        })


@app.route("/api/portfolio", methods=["POST"])
def get_portfolio():
    data = request.get_json() or {}
    codes: list[str] = data.get("codes", [])
    if len(codes) < 2:
        return jsonify({"success": False, "error": "至少需要选择 2 只股票"}), 400

    # ── 1. fetch price series for each code ──────────────────────────────────
    cl = _get("stock_list", ttl=86400)
    name_map: dict[str, str] = {s["代码"]: s["名称"] for s in cl["data"]} if cl else {}
    tc_map:   dict[str, str] = {s["代码"]: s["_tc"]  for s in cl["data"]} if cl else {}

    series: dict[str, tuple[list[str], list[float]]] = {}
    for code in codes:
        cached = _get(f"hist_{code}", ttl=3600)
        if cached and cached.get("success"):
            series[code] = (cached["dates"], cached["prices"])
            continue
        tc = tc_map.get(code, ("sh" if code.startswith("6") else "sz") + code)
        try:
            dates, closes = _fetch_kline_hfq(tc, years=10)
            if dates:
                series[code] = (dates, closes)
        except Exception:
            pass

    valid_codes = [c for c in codes if c in series]
    if len(valid_codes) < 2:
        return jsonify({"success": False,
                        "error": "有效历史数据不足，请确保所选股票有行情数据"}), 400

    # ── 2. intersect dates ────────────────────────────────────────────────────
    common_set = set(series[valid_codes[0]][0])
    for c in valid_codes[1:]:
        common_set &= set(series[c][0])
    common_dates = sorted(common_set)

    if len(common_dates) < 252:
        return jsonify({"success": False,
                        "error": f"共同交易日仅 {len(common_dates)} 天（不足一年），"
                                 "请选择上市时间相近的股票"}), 400

    # ── 3. normalize each stock to 1.0 at common start ───────────────────────
    normalized: dict[str, list[float]] = {}
    for code in valid_codes:
        dates, closes = series[code]
        idx_map = {d: i for i, d in enumerate(dates)}
        pts  = [closes[idx_map[d]] for d in common_dates]
        base = pts[0]
        normalized[code] = [round(p / base, 6) for p in pts]

    # ── 4. equal-weight portfolio ─────────────────────────────────────────────
    n = len(valid_codes)
    portfolio = [
        round(sum(normalized[c][i] for c in valid_codes) / n, 6)
        for i in range(len(common_dates))
    ]

    # ── 5. regression (mirrors _compute_regression_payload) ───────────────────
    arr   = np.array(portfolio, dtype=float)
    log_y = np.log(arr)
    N     = len(log_y)
    x_all = np.arange(N, dtype=float)

    s10, b10, r2_10 = _ols(x_all, log_y)
    reg_10y = np.exp(s10 * x_all + b10).tolist()

    CUT_5Y      = 5 * 365 + 10
    boundary_5y = (datetime.now() - timedelta(days=CUT_5Y)).strftime("%Y-%m-%d")
    split_idx   = next((i for i, d in enumerate(common_dates) if d >= boundary_5y), N // 2)

    x_5y  = x_all[split_idx:]
    ly_5y = log_y[split_idx:]
    s5, b5, r2_5 = _ols(x_5y, ly_5y)
    reg_5y = np.exp(s5 * x_all + b5).tolist()

    annual_growth_10 = (np.exp(s10 * 252) - 1.0) * 100
    annual_growth_5  = (np.exp(s5  * 252) - 1.0) * 100

    res_5y = ly_5y - (s5 * x_5y + b5)
    p10    = float(np.percentile(res_5y, 10))
    p90    = float(np.percentile(res_5y, 90))

    lower_band = np.exp(s5 * x_all + b5 + p10)
    upper_band = np.exp(s5 * x_all + b5 + p90)

    cur_log_dev = float(log_y[-1] - (s5 * x_all[-1] + b5))
    channel_w   = p90 - p10
    channel_pos = float((cur_log_dev - p10) / channel_w * 100) if channel_w > 0 else 50.0

    prices_5y    = arr[split_idx:]
    rolling_peak = np.maximum.accumulate(prices_5y)
    max_drawdown = float(np.min((prices_5y - rolling_peak) / rolling_peak)) * 100

    cur_price        = float(arr[-1])
    lower_band_price = float(lower_band[-1])
    upper_band_price = float(upper_band[-1])

    # ── 6. per-component R² (use cache when available) ────────────────────────
    component_stats = []
    for code in valid_codes:
        entry = {"code": code, "name": name_map.get(code, code)}
        cached = _get(f"hist_{code}", ttl=3600)
        if cached and cached.get("success"):
            entry.update({
                "r2_10": round(cached["stats_10y"]["r_squared"] * 100, 1),
                "r2_5":  round(cached["stats_5y"]["r_squared"]  * 100, 1),
                "g10":   round(cached["stats_10y"]["annual_growth"], 2),
                "g5":    round(cached["stats_5y"]["annual_growth"],  2),
            })
        else:
            # compute on-the-fly from raw series
            dates_c, closes_c = series[code]
            arr_c = np.array(closes_c, dtype=float)
            log_c = np.log(arr_c)
            N_c   = len(log_c)
            x_c   = np.arange(N_c, dtype=float)
            s10c, _, r2_10c = _ols(x_c, log_c)
            entry["g10"]   = round((np.exp(s10c * 252) - 1.0) * 100, 2)
            entry["r2_10"] = round(r2_10c * 100, 1)
            sp_c = next(
                (i for i, d in enumerate(dates_c) if d >= boundary_5y), N_c // 2
            )
            s5c, _, r2_5c = _ols(x_c[sp_c:], log_c[sp_c:])
            entry["g5"]   = round((np.exp(s5c * 252) - 1.0) * 100, 2)
            entry["r2_5"] = round(r2_5c * 100, 1)
        component_stats.append(entry)

    return jsonify({
        "success":       True,
        "dates":         common_dates,
        "portfolio":     portfolio,
        "components":    normalized,
        "reg_10y":       [round(v, 4) for v in reg_10y],
        "reg_5y":        [round(v, 4) for v in reg_5y],
        "lower_band":    [round(v, 4) for v in lower_band.tolist()],
        "upper_band":    [round(v, 4) for v in upper_band.tolist()],
        "split_5y_date": boundary_5y,
        "split_5y_idx":  split_idx,
        "stats_10y": {
            "annual_growth": round(annual_growth_10, 2),
            "r_squared":     round(r2_10, 4),
            "total_days":    N,
        },
        "stats_5y": {
            "annual_growth": round(annual_growth_5, 2),
            "r_squared":     round(r2_5, 4),
            "total_days":    N - split_idx,
        },
        "band_stats": {
            "channel_pos":      round(max(0.0, min(200.0, channel_pos)), 1),
            "max_drawdown":     round(max_drawdown, 1),
            "lower_band_price": round(lower_band_price, 4),
            "upper_band_price": round(upper_band_price, 4),
            "stop_loss_pct":    round((lower_band_price - cur_price) / cur_price * 100, 1),
            "take_profit_pct":  round((upper_band_price - cur_price) / cur_price * 100, 1),
        },
        "component_stats": component_stats,
    })


@app.route("/api/analyze/<code>")
def analyze_stock(code: str):
    """Stream an AI analysis of the stock's fundamentals via Gemini on ModelHub."""
    try:
        from openai import AzureOpenAI as _AzureOpenAI
    except ImportError:
        return jsonify({"error": "请先安装 openai 库：pip install openai"}), 500

    api_key = os.environ.get("MODELHUB_API_KEY", "0wEyJRUC23zEedqDxEAtc81kZmoS5W9p")

    hist_data = _get(f"hist_{code}", ttl=3600)
    if not hist_data:
        return jsonify({"error": "请先在图表页加载该股票数据"}), 400

    pe_data = _get(f"pe_{code}", ttl=3600)

    # Resolve stock name
    cl   = _get("stock_list", ttl=86400)
    name = code
    if cl:
        for s in cl["data"]:
            if s["代码"] == code:
                name = s["名称"]
                break

    s5  = hist_data.get("stats_5y",  {})
    s10 = hist_data.get("stats_10y", {})
    d5  = pe_data.get("decomposition_5y") if pe_data else None
    d10 = pe_data.get("decomposition")    if pe_data else None
    ps  = pe_data.get("pe_stats", {})     if pe_data else {}

    # ── Build context block ───────────────────────────────────────────────────
    ctx: list[str] = [f"【{name}（{code}）A股上市公司】"]
    ctx.append(
        f"股价趋势：10年年化 {s10.get('annual_growth', 0):.1f}%/年"
        f"（R²={s10.get('r_squared', 0)*100:.0f}%），"
        f"5年年化 {s5.get('annual_growth', 0):.1f}%/年"
        f"（R²={s5.get('r_squared', 0)*100:.0f}%）"
    )
    if ps:
        ctx.append(
            f"当前PE(TTM)：{ps.get('current', 'N/A')}x，"
            f"历史{ps.get('percentile', 'N/A')}%分位"
            f"（P10={ps.get('p10')}x  中位={ps.get('median')}x  P90={ps.get('p90')}x）"
        )
    if d5:
        ctx.append(
            f"\n近5年收益拆解（{d5['period_start'][:7]} → {d5['period_end'][:7]}）："
            f"\n  后复权总涨幅 {d5['hfq_total_return']:+.1f}%"
            f"\n  EPS+股息累计 {d5['eps_div_return']:+.1f}%"
            f"\n  PE变化 {d5['pe_start']}x → {d5['pe_end']}x（{d5['pe_return']:+.1f}%）"
        )
    if d10:
        ctx.append(
            f"\n近10年收益拆解（{d10['period_start'][:7]} → {d10['period_end'][:7]}）："
            f"\n  后复权总涨幅 {d10['hfq_total_return']:+.1f}%"
            f"\n  EPS+股息累计 {d10['eps_div_return']:+.1f}%"
            f"\n  PE变化 {d10['pe_start']}x → {d10['pe_end']}x（{d10['pe_return']:+.1f}%）"
        )

    pe_chg = f"从{d5['pe_start']}x变化到{d5['pe_end']}x" if d5 else "发生了变化"
    eps_s  = f"{d5['eps_div_return']:+.1f}%" if d5 else "N/A"

    prompt = f"""你是A股资深分析师。量化数据如下：

{chr(10).join(ctx)}

请搜索 {name}（{code}）最新财报、行业动态，然后用中文简洁回答以下四题
（每题1-2句，总字数不超过250字，直接回答，不重复题目）：

**① 近5年估值变化主因**：PE{pe_chg}，核心驱动是什么？

**② EPS+股息{eps_s}的原因**：增长或下滑的关键因素？

**③ 趋势可持续性**：哪些可延续，哪些是一次性的，主要风险？

**④ 未来1-3年展望**：当前估值分位下的催化剂或下行风险？

风格：专业简洁，不加免责声明。
输出格式：使用 Markdown（**加粗**、空行分段），不要输出任何 HTML 标签。"""

    cache_key = f"analysis_v3_{code}"   # v3 = no HTML, markdown only
    cached    = _get(cache_key, ttl=7200)

    @stream_with_context
    def generate():
        if cached:
            yield f"data: {json.dumps({'text': cached, 'cached': True})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        buf: list[str] = []
        try:
            endpoint = os.environ.get(
                "MODELHUB_ENDPOINT",
                "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl",
            )
            model = os.environ.get("MODELHUB_MODEL", "gemini-3-pro-preview-new")

            client = _AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-03-01-preview",
            )

            stream = client.chat.completions.create(
                model=model,
                stream=True,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "google_search"}],
                tool_choice="auto",
                max_tokens=10000,
                extra_headers={"X-TT-LOGID": f"dp-finder-{code}"},
            )

            in_search   = False
            search_n    = 0
            tc_args     = {}   # tool_call index → accumulated arguments string
            fin_reason  = None

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta  = chunk.choices[0].delta
                reason = chunk.choices[0].finish_reason

                # ── tool call (google_search) chunks ─────────────────────────
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_args:
                            # First chunk for this tool call → show spinner
                            tc_args[idx] = ""
                            if not in_search:
                                in_search = True
                                search_n += 1
                                yield f"data: {json.dumps({'status': f'🔍 搜索中 ({search_n})…'})}\n\n"
                        # Accumulate arguments JSON to extract query string
                        if tc.function and tc.function.arguments:
                            tc_args[idx] += tc.function.arguments
                            try:
                                args = json.loads(tc_args[idx])
                                q = args.get("query", "")
                                if q:
                                    yield f"data: {json.dumps({'status': f'🔍 ({search_n}) {q}'})}\n\n"
                            except (json.JSONDecodeError, AttributeError):
                                pass  # still receiving partial JSON

                # ── text content ─────────────────────────────────────────────
                if delta.content:
                    if in_search:
                        in_search = False
                        yield f"data: {json.dumps({'status': ''})}\n\n"
                    buf.append(delta.content)
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

                if reason in ("stop", "length"):
                    fin_reason = reason
                    break

            # Only cache complete responses; skip caching truncated ones
            if fin_reason == "stop":
                _set(cache_key, "".join(buf))
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/analyze_chart/<code>", methods=["POST"])
def analyze_chart(code: str):
    """Stream a phase-analysis of the stock chart image via Gemini Vision on ModelHub."""
    try:
        from openai import AzureOpenAI as _AzureOpenAI
    except ImportError:
        return jsonify({"error": "请先安装 openai 库：pip install openai"}), 500

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "缺少 image 字段"}), 400

    api_key = os.environ.get("MODELHUB_API_KEY", "0wEyJRUC23zEedqDxEAtc81kZmoS5W9p")

    # Resolve stock name
    cl = _get("stock_list", ttl=86400)
    name = code
    if cl:
        for s in cl["data"]:
            if s["代码"] == code:
                name = s["名称"]
                break

    # Get actual chart date range from cached hist data
    hist_data = _get(f"hist_{code}", ttl=3600)
    chart_start, chart_end = "", ""
    if hist_data:
        dates = hist_data.get("dates", [])
        if dates:
            chart_start = dates[0][:7]   # YYYY-MM
            chart_end   = dates[-1][:7]  # YYYY-MM
    date_range_hint = (
        f"图中数据的实际时间范围是 {chart_start} 至 {chart_end}。"
        if chart_start and chart_end else ""
    )

    cache_key = f"phase_v5_{code}"       # v5 = full-range coverage required
    cached = _get(cache_key, ttl=7200)

    prompt = (
        f"你是A股资深分析师。图中是{name}（{code}）的后复权股价走势图及PE(TTM)历史曲线。"
        f"{date_range_hint}\n\n"
        "请将图中完整历史从头到尾划分为3到5个阶段，规则：\n"
        "1. 第一个阶段的start必须等于图中最早数据的年月\n"
        "2. 最后一个阶段的end必须等于图中最新数据的年月\n"
        "3. 相邻阶段必须无缝衔接（上一阶段end的下月 = 下一阶段start），不允许有时间空白\n\n"
        "输出裸JSON数组（禁止使用```代码块，直接以[开头）：\n"
        '[{"num":1,"start":"YYYY-MM","end":"YYYY-MM","title":"标题",'
        '"reason":"起因","process":"经过","ending":"终结"}, ...]\n\n'
        "要求：start/end格式严格YYYY-MM；每个阶段三字段合计不超过200字；输出中文。"
    )

    @stream_with_context
    def generate():
        if cached:
            yield f"data: {json.dumps({'text': cached, 'cached': True})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        buf: list[str] = []
        try:
            endpoint = os.environ.get(
                "MODELHUB_ENDPOINT",
                "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl",
            )
            model = os.environ.get("MODELHUB_MODEL", "gemini-3-pro-preview-new")

            client = _AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-03-01-preview",
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_b64},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            stream = client.chat.completions.create(
                model=model,
                stream=True,
                messages=messages,
                tools=[{"type": "google_search"}],
                tool_choice="auto",
                max_tokens=10000,
                extra_headers={"X-TT-LOGID": f"dp-finder-phase-{code}"},
            )

            in_search  = False
            search_n   = 0
            tc_args: dict = {}
            fin_reason = None

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reason = chunk.choices[0].finish_reason

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_args:
                            tc_args[idx] = ""
                            if not in_search:
                                in_search = True
                                search_n += 1
                                yield f"data: {json.dumps({'status': f'🔍 搜索中 ({search_n})…'})}\n\n"
                        if tc.function and tc.function.arguments:
                            tc_args[idx] += tc.function.arguments
                            try:
                                args = json.loads(tc_args[idx])
                                q = args.get("query", "")
                                if q:
                                    yield f"data: {json.dumps({'status': f'🔍 ({search_n}) {q}'})}\n\n"
                            except (json.JSONDecodeError, AttributeError):
                                pass

                if delta.content:
                    if in_search:
                        in_search = False
                        yield f"data: {json.dumps({'status': ''})}\n\n"
                    buf.append(delta.content)
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

                if reason in ("stop", "length"):
                    fin_reason = reason
                    break

            # Only cache complete responses; skip caching truncated ones
            if fin_reason == "stop":
                _set(cache_key, "".join(buf))
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
