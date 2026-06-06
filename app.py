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

# 自动加载 .env 文件（本地开发用，不提交到 git）
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import baostock as bs
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True   # 模板修改后无需重启服务器

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

# ── TTL cache（优先 Redis，降级内存）────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()
_redis = None

def _init_redis():
    global _redis
    url = os.environ.get("REDIS_URL")
    if not url:
        return
    try:
        import redis as redis_lib
        client = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=3)
        client.ping()
        _redis = client
        print(f"[cache] Redis connected: {url[:30]}...")
    except Exception as e:
        print(f"[cache] Redis unavailable, using memory: {e}")

_init_redis()


def _redis_cleanup():
    """启动时清理占空间的大型 key（kline 历史、旧趋势列表）。"""
    if not _redis:
        return
    try:
        deleted = 0
        for pattern in ("kline_hist_*", "kline_recent_*", "trend_stock_list_v3",
                        "us_kline_v2_*", "us_hfq_k_*"):
            cursor = 0
            while True:
                cursor, keys = _redis.scan(cursor, match=pattern, count=200)
                if keys:
                    _redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
        if deleted:
            print(f"[cache] Redis cleanup: deleted {deleted} large/stale keys")
    except Exception as e:
        print(f"[cache] Redis cleanup error: {e}")

_redis_cleanup()


def _get(key: str, ttl: int):
    if _redis:
        try:
            raw = _redis.get(key)
            if raw is not None:
                entry = json.loads(raw)
                if (time.time() - entry["ts"]) < ttl:
                    return entry["data"]
                _redis.delete(key)
                return None
        except Exception:
            pass
    with _cache_lock:
        entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


_REDIS_MAX_BYTES = 30 * 1024   # 超过 30 KB 的大数据只存内存，不占 Redis 空间

def _set(key: str, data):
    if _redis:
        try:
            payload = json.dumps({"data": data, "ts": time.time()})
            if len(payload) <= _REDIS_MAX_BYTES:
                _redis.setex(key, 86400 * 2, payload)  # 2 天 TTL，免费层不超限
                return
        except Exception:
            pass
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

def _fetch_hk_consensus_yf(code: str) -> dict:
    """Analyst consensus for HK stocks via yfinance.
    EPS estimates are in the financial reporting currency (CNY/USD); we convert
    to HKD using trailingEps as the exchange-rate anchor, same as PE calc.
    Returns the same dict shape as _fetch_consensus_eps."""
    cache_key = f"hk_consensus_yf_v1_{code}"
    cached = _get(cache_key, ttl=3600 * 12)
    if cached is not None:
        return cached

    empty = {"forecasts": [], "org_num": None, "buy_num": None,
             "add_num": None, "target_high": None, "target_low": None}
    try:
        import yfinance as yf, math as _m
        from datetime import datetime as _dt

        numeric = code.lstrip("0") or "0"
        t       = yf.Ticker(numeric.zfill(4) + ".HK")
        info    = t.info

        # FX scale: financial currency → HKD (same as _compute_pe_payload_hk)
        trailing_hkd = float(info.get("trailingEps") or 0)
        ann          = t.income_stmt
        eps_row = (ann.loc["Diluted EPS"] if "Diluted EPS" in ann.index
                   else ann.loc["Basic EPS"] if "Basic EPS" in ann.index else None)
        # sort_index() ensures ascending order; iloc[-1] = most recent annual EPS
        recent_fccy  = float(eps_row.dropna().sort_index().iloc[-1]) if eps_row is not None else 0
        fx = (trailing_hkd / recent_fccy
              if trailing_hkd > 0 and recent_fccy > 0 else 1.0)

        # Annual EPS estimates from yfinance (0y = current FY, +1y = next FY)
        ee = t.earnings_estimate
        forecasts = []
        cur_year  = _dt.now().year
        for period, offset in [("0y", 0), ("+1y", 1)]:
            if period not in ee.index:
                continue
            row = ee.loc[period]
            avg_fccy = float(row.get("avg") or 0)
            if avg_fccy <= 0 or _m.isnan(avg_fccy):
                continue
            eps_hkd = round(avg_fccy * fx, 2)
            n_analysts = int(row.get("numberOfAnalysts") or 0)
            forecasts.append({
                "year":           cur_year + offset,
                "year_mark":      "E",
                "eps":            eps_hkd,
                "forward_pe":     None,
                "n_analysts":     n_analysts,
            })

        # Analyst price targets (already in HKD)
        pt = t.analyst_price_targets
        target_high = round(float(pt.get("high") or 0), 2) or None
        target_low  = round(float(pt.get("low")  or 0), 2) or None
        target_mean = round(float(pt.get("mean") or 0), 2) or None

        # Recommendation summary for buy/add counts
        buy_num = add_num = None
        try:
            rs = t.recommendations_summary
            if rs is not None and not rs.empty:
                latest = rs.iloc[0]
                buy_num = int(latest.get("strongBuy", 0) + latest.get("buy", 0))
                add_num = int(latest.get("hold", 0))
        except Exception:
            pass

        org_num = max((f.get("n_analysts") or 0) for f in forecasts) if forecasts else None

        result = {
            "forecasts":   forecasts,
            "org_num":     org_num,
            "buy_num":     buy_num,
            "add_num":     add_num,
            "target_high": target_high,
            "target_low":  target_low,
            "target_mean": target_mean,
        }
        _set(cache_key, result)
        return result
    except Exception as exc:
        import sys; print(f"[hk_consensus_yf] {code}: {exc}", file=sys.stderr)
        _set(cache_key, empty)
        return empty


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


def _fetch_a_stock_list(threshold_yi: float = 0.0) -> list[dict]:
    """Fetch A-share list (沪深主板 + 创业板) from Eastmoney push2 clist.

    Returns list of dicts sorted by market cap descending, each with:
      代码, 名称, 行业, 市值亿, 最新价, 涨跌幅, pe, _tc
    Cached for 24 h.
    """
    cache_key = "a_stock_list_v1"
    cached = _get(cache_key, ttl=86400)
    if cached is not None:
        return cached

    EM_URL    = "https://push2delay.eastmoney.com/api/qt/clist/get"
    EM_PARAMS = {
        "po": "1", "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2", "invt": "2", "fid": "f20",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f2,f3,f9,f12,f14,f20,f100",
    }

    def _parse_items(diff: list) -> list[dict]:
        out = []
        for item in diff:
            code = str(item.get("f12") or "").strip()
            if not code:
                continue
            try:
                mktcap = float(item.get("f20") or 0) / 1e8
            except (TypeError, ValueError):
                continue
            if mktcap <= 0:
                continue
            try:
                price = round(float(item.get("f2") or 0), 2)
            except (TypeError, ValueError):
                price = 0.0
            try:
                chg = round(float(item.get("f3") or 0), 2)
            except (TypeError, ValueError):
                chg = 0.0
            try:
                pe_raw = float(item.get("f9") or 0)
                pe = round(pe_raw, 1) if -9999 < pe_raw < 9999 else 0.0
            except (TypeError, ValueError):
                pe = 0.0
            market = "sh" if code.startswith("6") else "sz"
            out.append({
                "代码":   code,
                "名称":   str(item.get("f14") or ""),
                "行业":   str(item.get("f100") or "").replace("Ⅱ", ""),
                "市值亿": round(mktcap, 1),
                "最新价": price,
                "涨跌幅": chg,
                "pe":     pe,
                "_tc":    f"{market}{code}",
            })
        return out

    def _fetch_page(page_no: int) -> list[dict]:
        try:
            r = _em_sess.get(EM_URL, params={**EM_PARAMS, "pn": str(page_no), "pz": "100"},
                             timeout=15)
            diff = (r.json().get("data") or {}).get("diff") or []
            return _parse_items(diff)
        except Exception:
            return []

    result: list[dict] = []
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 第1页：获取总数
        r0 = _em_sess.get(EM_URL, params={**EM_PARAMS, "pn": "1", "pz": "100"}, timeout=15)
        data0 = r0.json().get("data") or {}
        total = int(data0.get("total") or 0)
        result.extend(_parse_items(data0.get("diff") or []))

        if total > 100:
            remaining_pages = list(range(2, (total // 100) + 2))
            # 8 个并发，适当限速避免被封
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_fetch_page, p): p for p in remaining_pages}
                for fut in as_completed(futures):
                    result.extend(fut.result())

    except Exception:
        pass

    result.sort(key=lambda x: x["市值亿"], reverse=True)
    if result:
        _set(cache_key, result)
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


def _fetch_kline_range(tc_code: str, start: str, end: str) -> tuple[list[str], list[float]]:
    """从腾讯接口拉取指定日期范围的后复权日线，内部按900天分块请求。"""
    base = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param={code},day,{s},{e},700,hfq"
    )
    boundaries: list[str] = []
    t = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while t < end_dt:
        boundaries.append(t.strftime("%Y-%m-%d"))
        t += timedelta(days=900)
    boundaries.append(end)

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
            sd   = r.json()["data"][tc_code]
            rows = sd.get("hfqday") or sd.get("qfqday") or sd.get("day") or []
        except (KeyError, TypeError):
            continue
        for row in rows:
            d = row[0]
            if not dates or d > dates[-1]:
                dates.append(d)
                closes.append(float(row[2]))
    return dates, closes


def _fetch_kline_hfq(tc_code: str, years: int = 10) -> tuple[list[str], list[float]]:
    """冷热分离缓存：历史段缓存50天，近期段缓存24小时。

    cutoff（历史段结束日期）存入历史段 payload，近期段始终从该 cutoff 开始拉取，
    保证两段无缝衔接，不会因 cutoff 随时间漂移而出现空洞。
    历史段过期重建时，同步清除近期段缓存使其重新拉取。
    """
    today      = datetime.now()
    end        = today.strftime("%Y-%m-%d")
    hist_start = (today - timedelta(days=years * 365 + 10)).strftime("%Y-%m-%d")

    # ── 历史段（10年 → cutoff），TTL 50天 ────────────────────────────────────
    hist = _get(f"kline_hist_{tc_code}", ttl=86400 * 50)
    # 若缓存中记录的 hist_start 比当前请求的更晚（说明是旧的短周期缓存），强制重取。
    # 注：比较的是"请求时的起始日期"而非 dates[0]，避免新上市股因上市日期
    # 晚于阈值而被反复清缓存（dates[0] = 上市日 是正确数据，不应触发失效）。
    if hist:
        cached_start = hist.get("hist_start", "2022-01-01")  # 旧缓存无此字段，默认视为4年
        if cached_start > hist_start:
            hist = None
    if not hist:
        cutoff = (today - timedelta(days=50)).strftime("%Y-%m-%d")
        h_dates, h_closes = _fetch_kline_range(tc_code, hist_start, cutoff)
        if h_dates:
            hist = {"dates": h_dates, "closes": h_closes, "cutoff": cutoff,
                    "hist_start": hist_start}
            _set(f"kline_hist_{tc_code}", hist)
            # 历史段更新，清除近期段让其从新 cutoff 重拉
            if _redis:
                try:
                    _redis.delete(f"kline_recent_{tc_code}")
                except Exception:
                    pass
            else:
                with _cache_lock:
                    _cache.pop(f"kline_recent_{tc_code}", None)

    # ── 近期段（cutoff → 今天），TTL 24小时 ──────────────────────────────────
    # 始终从历史段存储的 cutoff 开始，而不是重新算"今天-50天"
    recent_start = (hist or {}).get("cutoff") or (today - timedelta(days=55)).strftime("%Y-%m-%d")
    recent = _get(f"kline_recent_{tc_code}", ttl=86400)
    if not recent:
        r_dates, r_closes = _fetch_kline_range(tc_code, recent_start, end)
        if r_dates:
            recent = {"dates": r_dates, "closes": r_closes}
            _set(f"kline_recent_{tc_code}", recent)

    # ── 合并，按日期去重 ──────────────────────────────────────────────────────
    dates:  list[str]   = list(hist["dates"])   if hist   else []
    closes: list[float] = list(hist["closes"])  if hist   else []
    seen = set(dates)
    if recent:
        for d, c in zip(recent["dates"], recent["closes"]):
            if d not in seen:
                dates.append(d)
                closes.append(c)
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


def _fetch_hk_annual_eps_em(secucode: str) -> list[tuple[str, float]]:
    """Fetch annual diluted EPS from Eastmoney HK F10 for a HK stock.
    secucode format: '00700.HK'.  Returns [(date_str, eps_fccy), ...] oldest→newest,
    in the stock's financial reporting currency (CNY, USD, HKD, etc.)."""
    try:
        r = _sess.get(
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            params={
                "reportName": "RPT_HKF10_FN_INCOME",
                "columns":    "REPORT_DATE,AMOUNT",
                # both 001027003 (int'l GAAP) and 004027003 (HK GAAP) = diluted EPS
                "filter":     f'(SECUCODE="{secucode}")'
                              '(STD_ITEM_CODE in ("001027003","004027003"))'
                              '(REPORT_TYPE="年报")',
                "pageSize":   30,
                "sortColumns": "REPORT_DATE",
                "sortTypes":   1,
            },
            timeout=8,
        )
        rows = (r.json().get("result") or {}).get("data") or []
        return [(row["REPORT_DATE"][:10], float(row["AMOUNT"]))
                for row in rows if row.get("AMOUNT") is not None]
    except Exception:
        return []


def _compute_pe_payload_hk(code: str) -> dict | None:
    """Compute historical PE for HK stocks.
    EPS source: Eastmoney HK F10 (up to 20 years) with yfinance trailingEps as
    FX-to-HKD scale reference.  Falls back to yfinance income_stmt if EM fails.
    Returns the same payload structure as _compute_pe_payload. Cached 1 day."""
    cache_key = f"pe_hk_v2_{code}"
    cached = _get(cache_key, ttl=86400)
    if cached:
        return cached
    try:
        import yfinance as yf, math as _math
        numeric  = code.lstrip("0") or "0"
        yf_tick  = numeric.zfill(4) + ".HK"
        t        = yf.Ticker(yf_tick)

        # ── 1. Annual EPS (financial reporting currency) ──────────────────────
        em_eps = _fetch_hk_annual_eps_em(f"{numeric.zfill(5)}.HK")
        if len(em_eps) >= 2:
            eps_dates = [d for d, _ in em_eps]
            eps_vals  = [v for _, v in em_eps]
        else:
            # Fallback: yfinance income_stmt (only 4-5 years)
            ann = t.income_stmt
            if "Diluted EPS" in ann.index:
                eps_s = ann.loc["Diluted EPS"].dropna().sort_index()
            elif "Basic EPS" in ann.index:
                eps_s = ann.loc["Basic EPS"].dropna().sort_index()
            else:
                return None
            if len(eps_s) < 2:
                return None
            eps_dates = [str(d)[:10] for d in eps_s.index]
            eps_vals  = [float(v) for v in eps_s.values]

        # ── 2. FX scale: convert financial-currency EPS → HKD ─────────────────
        # yfinance trailingEps is already in the trading currency (HKD).
        trailing_eps_hkd = float(t.info.get("trailingEps") or 0)
        recent_eps_fccy  = next((v for v in reversed(eps_vals) if v and v > 0), 0)
        fx_scale = (trailing_eps_hkd / recent_eps_fccy
                    if trailing_eps_hkd > 0 and recent_eps_fccy > 0 else 1.0)
        eps_vals_hkd = [v * fx_scale for v in eps_vals]

        # ── 3. Actual (non-adjusted) daily prices ─────────────────────────────
        hist = t.history(period="max", auto_adjust=False, actions=False)
        if hist.empty:
            return None

        pe_dates: list[str]   = []
        pe_vals:  list[float] = []
        for ts, row in hist.iterrows():
            price = float(row["Close"])
            if _math.isnan(price) or price <= 0:
                continue
            date_str = str(ts)[:10]
            # most recent annual EPS (HKD) at or before this price date
            eps_v = None
            for d, v in zip(eps_dates, eps_vals_hkd):
                if d <= date_str and v > 0 and not _math.isnan(v):
                    eps_v = v
            if eps_v is None:
                continue
            pe = round(price / eps_v, 2)
            if 0 < pe < 500:
                pe_dates.append(date_str)
                pe_vals.append(pe)

        if len(pe_vals) < 50:
            return None

        arr_pe        = np.array(pe_vals, dtype=float)
        pe_current    = float(pe_vals[-1])
        pe_percentile = float(np.mean(arr_pe < pe_current) * 100)
        payload: dict = {
            "pe_dates":   pe_dates,
            "pe_vals":    pe_vals,
            "loss_ranges": [],
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

        # ── 收益来源拆解（EPS增长 vs PE扩张）─────────────────────────────────
        main = _get(f"hist_v4_{code}", ttl=86400)
        if main and main.get("success"):
            price_dates = main["dates"]
            prices_hfq  = main["prices"]
            pe_dict = dict(zip(pe_dates, pe_vals))
            last_ov = next((d for d in reversed(price_dates) if d in pe_dict), None)

            def _decomp_hk(start_idx: int) -> dict | None:
                start_pd   = price_dates[start_idx]
                first_pe_d = next((d for d in pe_dates if d >= start_pd), None)
                if not first_pe_d or not last_ov or first_pe_d >= last_ov:
                    return None
                if first_pe_d not in pe_dict or last_ov not in pe_dict:
                    return None
                p_s = prices_hfq[start_idx]
                p_e = prices_hfq[price_dates.index(last_ov)]
                pe_s, pe_e = pe_dict[first_pe_d], pe_dict[last_ov]
                hfq_ret     = p_e / p_s
                pe_ret      = pe_e / pe_s
                eps_div_ret = hfq_ret / pe_ret
                pl = float(np.log(pe_ret))
                el = float(np.log(eps_div_ret))
                ad = abs(pl) + abs(el)
                return {
                    "period_start":        first_pe_d,
                    "period_end":          last_ov,
                    "pe_start":            round(pe_s, 1),
                    "pe_end":              round(pe_e, 1),
                    "hfq_total_return":    round((hfq_ret     - 1) * 100, 1),
                    "pe_return":           round((pe_ret      - 1) * 100, 1),
                    "eps_div_return":      round((eps_div_ret - 1) * 100, 1),
                    "pe_contrib_pct":      round(pl / ad * 100, 1) if ad > 0.001 else 0.0,
                    "eps_div_contrib_pct": round(el / ad * 100, 1) if ad > 0.001 else 0.0,
                }

            s10 = next((i for i, d in enumerate(price_dates) if d in pe_dict), None)
            if s10 is not None:
                d10 = _decomp_hk(s10)
                if d10: payload["decomposition"] = d10

            s5 = main.get("split_5y_idx", 0)
            if s5 and s5 < len(price_dates):
                d5 = _decomp_hk(s5)
                if d5: payload["decomposition_5y"] = d5

        _set(cache_key, payload)
        return payload
    except Exception as exc:
        import sys; print(f"[pe_hk] {code}: {exc}", file=sys.stderr)
        return None


def _fetch_us_eps_edgar(ticker: str) -> list[tuple[str, float]]:
    """Fetch annual diluted EPS from SEC EDGAR XBRL API.
    Returns [(end_date, eps), ...] oldest→newest, unadjusted (same basis as
    yfinance auto_adjust=False prices).  Cached 7 days.
    Uses fp='FY' + fy-grouping to exclude within-year cumulative rows."""
    ck = f"edgar_eps_v1_{ticker}"
    cached = _get(ck, ttl=86400 * 7)
    if cached:
        return cached
    try:
        # Step 1: ticker → CIK (cached 30 days)
        cik_map = _get("edgar_cik_map_v1", ttl=86400 * 30)
        if not cik_map:
            r = _sess.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "stockscreener contact@example.com"},
                timeout=15,
            )
            cik_map = {v["ticker"]: str(v["cik_str"]).zfill(10)
                       for v in r.json().values()}
            _set("edgar_cik_map_v1", cik_map)

        # Some tickers use dots in EDGAR (e.g., BRK-B → BRK-B)
        cik = cik_map.get(ticker) or cik_map.get(ticker.replace("-", "."))
        if not cik:
            return []

        # Step 2: fetch XBRL company facts
        r = _sess.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={"User-Agent": "stockscreener contact@example.com"},
            timeout=20,
        )
        facts = r.json()["facts"].get("us-gaap", {})
        eps_tag = facts.get("EarningsPerShareDiluted") or facts.get("EarningsPerShareBasic")
        if not eps_tag:
            return []

        rows = eps_tag["units"].get("USD/shares", [])

        # Step 3: 10-K fp=FY only; group by fy; keep latest end+filed per fy
        best: dict = {}
        for x in rows:
            if x.get("form") != "10-K" or x.get("fp") != "FY":
                continue
            fy = x.get("fy")
            if fy is None:
                continue
            prev = best.get(fy)
            if prev is None or (x["end"], x["filed"]) > (prev["end"], prev["filed"]):
                best[fy] = x

        result = [(x["end"], float(x["val"]))
                  for x in sorted(best.values(), key=lambda x: x["end"])
                  if x["val"] is not None]
        if result:
            _set(ck, result)
        return result
    except Exception as exc:
        import sys; print(f"[edgar_eps] {ticker}: {exc}", file=sys.stderr)
        return []


def _compute_pe_payload_us(code: str) -> dict | None:
    """Compute historical PE for US stocks.
    EPS source: SEC EDGAR XBRL (up to 15+ years) with yfinance as fallback.
    Prices: yfinance auto_adjust=False (actual, non-split-adjusted).
    Both EPS and price are unadjusted so PE is consistent across splits."""
    cache_key = f"pe_us_v2_{code}"
    cached = _get(cache_key, ttl=86400)
    if cached:
        return cached
    try:
        import yfinance as yf, math as _math

        # ── EPS: EDGAR first (15+ years), fall back to yfinance (4-5 years) ──
        edgar_eps = _fetch_us_eps_edgar(code)
        if len(edgar_eps) >= 2:
            eps_dates = [d for d, _ in edgar_eps]
            eps_vals  = [v for _, v in edgar_eps]
        else:
            t = yf.Ticker(code)
            ann = t.income_stmt
            if "Diluted EPS" in ann.index:
                eps_s = ann.loc["Diluted EPS"].dropna().sort_index()
            elif "Basic EPS" in ann.index:
                eps_s = ann.loc["Basic EPS"].dropna().sort_index()
            else:
                return None
            if len(eps_s) < 2:
                return None
            eps_dates = [str(d)[:10] for d in eps_s.index]
            eps_vals  = [float(v) for v in eps_s.values]

        t = yf.Ticker(code)

        # Use actual (unadjusted) prices. To handle stock splits consistently,
        # both price and EPS are divided by "splits occurring AFTER this date"
        # so they're always on the same per-share basis.
        # PE = (price / splits_after_price_date) / (eps / splits_after_eps_date)
        hist = t.history(period="max", auto_adjust=False, actions=True)
        if hist.empty:
            return None

        # Build sorted split list: date_str → ratio
        raw_splits = hist["Stock Splits"].dropna()
        raw_splits = raw_splits[raw_splits > 0].sort_index()
        splits_list = [(str(ts)[:10], float(ratio)) for ts, ratio in raw_splits.items()]

        def _splits_after(date_str: str) -> float:
            """Product of all split ratios that occurred STRICTLY AFTER date_str."""
            f = 1.0
            for sd, ratio in splits_list:
                if sd > date_str:
                    f *= ratio
            return f

        # Adjust EPS to "current shares" basis (divide by splits after eps date)
        adj_eps: dict[str, float] = {
            d: v / _splits_after(d) for d, v in zip(eps_dates, eps_vals)
        }
        adj_eps_dates = eps_dates  # same dates, values now normalised

        pe_dates: list[str]   = []
        pe_vals:  list[float] = []
        for ts, row in hist.iterrows():
            actual_price = float(row["Close"])
            if _math.isnan(actual_price) or actual_price <= 0:
                continue
            date_str = str(ts)[:10]
            # yfinance auto_adjust=False prices are already split-adjusted to
            # current-share basis; only EPS needs normalisation via _splits_after
            price = actual_price
            # Most recent adjusted EPS on or before this date
            eps_v = None
            for d in reversed(adj_eps_dates):
                if d <= date_str:
                    v = adj_eps[d]
                    if v > 0 and not _math.isnan(v):
                        eps_v = v
                    break
            if eps_v is None:
                continue
            pe = round(price / eps_v, 2)
            if 0 < pe < 500:
                pe_dates.append(date_str)
                pe_vals.append(pe)

        if len(pe_vals) < 50:
            return None

        arr_pe        = np.array(pe_vals, dtype=float)
        pe_current    = float(pe_vals[-1])
        pe_percentile = float(np.mean(arr_pe < pe_current) * 100)
        payload: dict = {
            "pe_dates":    pe_dates,
            "pe_vals":     pe_vals,
            "loss_ranges": [],
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

        # 收益来源拆解（EPS增长 vs PE扩张）
        main = _get(f"hist_v4_{code}", ttl=86400)
        if main and main.get("success"):
            price_dates = main["dates"]
            prices_hfq  = main["prices"]
            pe_dict = dict(zip(pe_dates, pe_vals))
            last_ov = next((d for d in reversed(price_dates) if d in pe_dict), None)

            def _decomp_us(start_idx: int) -> dict | None:
                start_pd   = price_dates[start_idx]
                first_pe_d = next((d for d in pe_dates if d >= start_pd), None)
                if not first_pe_d or not last_ov or first_pe_d >= last_ov:
                    return None
                if first_pe_d not in pe_dict or last_ov not in pe_dict:
                    return None
                p_s = prices_hfq[start_idx]
                p_e = prices_hfq[price_dates.index(last_ov)]
                pe_s, pe_e = pe_dict[first_pe_d], pe_dict[last_ov]
                hfq_ret     = p_e / p_s
                pe_ret      = pe_e / pe_s
                eps_div_ret = hfq_ret / pe_ret
                pl = float(np.log(pe_ret))
                el = float(np.log(eps_div_ret))
                ad = abs(pl) + abs(el)
                return {
                    "period_start":        first_pe_d,
                    "period_end":          last_ov,
                    "pe_start":            round(pe_s, 1),
                    "pe_end":              round(pe_e, 1),
                    "hfq_total_return":    round((hfq_ret     - 1) * 100, 1),
                    "pe_return":           round((pe_ret      - 1) * 100, 1),
                    "eps_div_return":      round((eps_div_ret - 1) * 100, 1),
                    "pe_contrib_pct":      round(pl / ad * 100, 1) if ad > 0.001 else 0.0,
                    "eps_div_contrib_pct": round(el / ad * 100, 1) if ad > 0.001 else 0.0,
                }

            s10 = next((i for i, d in enumerate(price_dates) if d in pe_dict), None)
            if s10 is not None:
                d10 = _decomp_us(s10)
                if d10: payload["decomposition"] = d10

            s5 = main.get("split_5y_idx", 0)
            if s5 and s5 < len(price_dates):
                d5 = _decomp_us(s5)
                if d5: payload["decomposition_5y"] = d5

        _set(cache_key, payload)
        return payload
    except Exception as exc:
        import sys; print(f"[pe_us] {code}: {exc}", file=sys.stderr)
        return None


def _fetch_us_consensus_yf(code: str) -> dict:
    """Analyst consensus for US stocks via yfinance. EPS already in USD."""
    cache_key = f"us_consensus_yf_v1_{code}"
    cached = _get(cache_key, ttl=3600 * 12)
    if cached is not None:
        return cached

    empty = {"forecasts": [], "org_num": None, "buy_num": None,
             "add_num": None, "target_high": None, "target_low": None}
    try:
        import yfinance as yf, math as _m
        from datetime import datetime as _dt

        t = yf.Ticker(code)
        ee = t.earnings_estimate
        forecasts = []
        cur_year  = _dt.now().year
        for period, offset in [("0y", 0), ("+1y", 1)]:
            if period not in ee.index:
                continue
            row = ee.loc[period]
            avg = float(row.get("avg") or 0)
            if avg <= 0 or _m.isnan(avg):
                continue
            forecasts.append({
                "year":       cur_year + offset,
                "year_mark":  "E",
                "eps":        round(avg, 2),
                "forward_pe": None,
                "n_analysts": int(row.get("numberOfAnalysts") or 0),
            })

        pt = t.analyst_price_targets
        target_high = round(float(pt.get("high") or 0), 2) or None
        target_low  = round(float(pt.get("low")  or 0), 2) or None
        target_mean = round(float(pt.get("mean") or 0), 2) or None

        buy_num = add_num = None
        try:
            rs = t.recommendations_summary
            if rs is not None and not rs.empty:
                latest = rs.iloc[0]
                buy_num = int(latest.get("strongBuy", 0) + latest.get("buy", 0))
                add_num = int(latest.get("hold", 0))
        except Exception:
            pass

        org_num = max((f.get("n_analysts") or 0) for f in forecasts) if forecasts else None
        result = {
            "forecasts":   forecasts,
            "org_num":     org_num,
            "buy_num":     buy_num,
            "add_num":     add_num,
            "target_high": target_high,
            "target_low":  target_low,
            "target_mean": target_mean,
        }
        _set(cache_key, result)
        return result
    except Exception as exc:
        import sys; print(f"[us_consensus_yf] {code}: {exc}", file=sys.stderr)
        _set(cache_key, empty)
        return empty


def _compute_pe_payload(tc_code: str, code: str) -> dict | None:
    """Fetch PE history + decompose price return into EPS-growth vs PE-expansion."""
    cache_key = f"pe_{code}"
    cached = _get(cache_key, ttl=86400)
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
    main = _get(f"hist_v4_{code}", ttl=86400)
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
    cache_key = f"hist_v4_{code}"
    cached = _get(cache_key, ttl=86400)
    if cached and cached.get("success"):
        return cached

    _is_us_code  = tc_code.startswith("us") and not tc_code[2:].isdigit()
    if _is_us_code:
        dates, closes = _fetch_us_kline_em(code, years=10)
    else:
        dates, closes = _fetch_kline_hfq(tc_code, years=10)
    if not dates:
        return None

    # Drop stocks listed less than 3 years ago (ETFs, HK, and US stocks exempt)
    _is_etf_code = any(e["代码"] == code for e in _MAJOR_ETFS)
    _is_hk_code  = tc_code.startswith("hk")
    if not _is_etf_code and not _is_hk_code and not _is_us_code:
        cutoff_3y = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        if dates[0] > cutoff_3y:
            return None

    if len(dates) < 60:   # need at least ~60 trading days for a meaningful regression
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

    # ── 滚动50日 ATR 止损带：每根K线往前50天取最高收盘价 + 平均绝对日收益率 ──────
    arr_c        = np.array(closes, dtype=float)
    n_c          = len(arr_c)
    sl_3atr_arr: list = [None] * n_c
    sl_4atr_arr: list = [None] * n_c
    for i in range(1, n_c):
        s_i    = max(0, i - 49)
        win    = arr_c[s_i : i + 1]
        max_c  = float(win.max())
        atr_w  = float(np.abs(win[1:] / win[:-1] - 1).mean()) * 100 if len(win) > 1 else 0.0
        sl_3atr_arr[i] = round(max_c * (1 - 3 * atr_w / 100), 4)
        sl_4atr_arr[i] = round(max_c * (1 - 4 * atr_w / 100), 4)
    # 末端标量（供 statsPanel 展示）
    atr_pct = round(float(np.abs(arr_c[-50:][1:] / arr_c[-50:][:-1] - 1).mean()) * 100, 2) \
              if n_c >= 2 else 0.0
    sl_3atr = sl_3atr_arr[-1] or 0.0
    sl_4atr = sl_4atr_arr[-1] or 0.0

    lower_band,    upper_band,    band_stats_5 = _band_stats(
        s5, b5, x_5y,  ly_5y,  x_all, arr[split_idx:],    cur_price)
    lower_band_3y, upper_band_3y, band_stats_3 = _band_stats(
        s3, b3, x_3y,  ly_3y,  x_all, arr[split_3y_idx:], cur_price)

    payload = {
        "success":        True,
        "atr_pct":        atr_pct,
        "sl_3atr":        sl_3atr,
        "sl_4atr":        sl_4atr,
        "sl_3atr_arr":    sl_3atr_arr,
        "sl_4atr_arr":    sl_4atr_arr,
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
    # ── 宽基 ──────────────────────────────────────────────────────────────────
    {"代码": "510300", "名称": "沪深300ETF(华泰)", "_tc": "sh510300", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "159919", "名称": "沪深300ETF(嘉实)", "_tc": "sz159919", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "510500", "名称": "中证500ETF(南方)", "_tc": "sh510500", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "159922", "名称": "中证500ETF(嘉实)", "_tc": "sz159922", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "512100", "名称": "中证1000ETF(南方)","_tc": "sh512100", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "510050", "名称": "上证50ETF(华夏)",  "_tc": "sh510050", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "512050", "名称": "A500ETF(华夏)",    "_tc": "sh512050", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "588000", "名称": "科创50ETF",        "_tc": "sh588000", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    {"代码": "159915", "名称": "创业板ETF",        "_tc": "sz159915", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "宽基ETF"},
    # ── 红利 / 策略 ───────────────────────────────────────────────────────────
    {"代码": "510880", "名称": "红利ETF(华泰)",    "_tc": "sh510880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515080", "名称": "中证红利ETF(招商)","_tc": "sh515080", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159905", "名称": "红利ETF(工银)",    "_tc": "sz159905", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515450", "名称": "红利低波50ETF",    "_tc": "sh515450", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 消费 / 食品 / 白酒 ───────────────────────────────────────────────────
    {"代码": "512690", "名称": "酒ETF(鹏华)",      "_tc": "sh512690", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515170", "名称": "食品饮料ETF(华夏)","_tc": "sh515170", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159928", "名称": "消费ETF(汇添富)",  "_tc": "sz159928", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 金融 / 银行 ───────────────────────────────────────────────────────────
    {"代码": "512800", "名称": "银行ETF(华宝)",    "_tc": "sh512800", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512880", "名称": "证券ETF",          "_tc": "sh512880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 医药 / 医疗 ───────────────────────────────────────────────────────────
    {"代码": "512010", "名称": "医药ETF(易方达)",  "_tc": "sh512010", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512170", "名称": "医疗ETF",          "_tc": "sh512170", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 科技 / 半导体 ────────────────────────────────────────────────────────
    {"代码": "512480", "名称": "半导体ETF",        "_tc": "sh512480", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159995", "名称": "芯片ETF",          "_tc": "sz159995", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 新能源 ────────────────────────────────────────────────────────────────
    {"代码": "515030", "名称": "新能源ETF",        "_tc": "sh515030", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159806", "名称": "新能源车ETF",      "_tc": "sz159806", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515790", "名称": "光伏ETF",          "_tc": "sh515790", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159892", "名称": "储能ETF",          "_tc": "sz159892", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 其他行业 ──────────────────────────────────────────────────────────────
    {"代码": "512660", "名称": "军工ETF",          "_tc": "sh512660", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "515220", "名称": "煤炭ETF",          "_tc": "sh515220", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "159611", "名称": "电力ETF",          "_tc": "sz159611", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512400", "名称": "有色ETF",          "_tc": "sh512400", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    {"代码": "512200", "名称": "房地产ETF(南方)",  "_tc": "sh512200", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "行业ETF"},
    # ── 大宗商品 ──────────────────────────────────────────────────────────────
    {"代码": "518880", "名称": "黄金ETF(华安)",    "_tc": "sh518880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159934", "名称": "黄金ETF(易方达)",  "_tc": "sz159934", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159812", "名称": "白银ETF",          "_tc": "sz159812", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159981", "名称": "石油ETF",          "_tc": "sz159981", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159985", "名称": "豆粕ETF",          "_tc": "sz159985", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    {"代码": "159980", "名称": "有色金属ETF",      "_tc": "sz159980", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "大宗商品ETF"},
    # ── 跨境 ──────────────────────────────────────────────────────────────────
    {"代码": "513660", "名称": "恒生ETF(华夏)",    "_tc": "sh513660", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513600", "名称": "恒生指数ETF(南方)","_tc": "sh513600", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513130", "名称": "恒生科技ETF",      "_tc": "sh513130", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513880", "名称": "日经225ETF(华安)", "_tc": "sh513880", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513520", "名称": "日经ETF(华夏)",    "_tc": "sh513520", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513500", "名称": "标普500ETF(博时)", "_tc": "sh513500", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "159612", "名称": "标普500ETF(国泰)", "_tc": "sz159612", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513100", "名称": "纳指ETF(华夏)",    "_tc": "sh513100", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "159659", "名称": "纳斯达克100ETF(招商)","_tc": "sz159659", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513050", "名称": "中概互联ETF(易方达)","_tc": "sh513050", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "159561", "名称": "德国ETF(嘉实)",    "_tc": "sz159561", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
    {"代码": "513730", "名称": "东南亚科技ETF",    "_tc": "sh513730", "pe": 0, "市值亿": 9999, "is_etf": True, "industry_board": "跨境ETF"},
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


# ── US ETFs ───────────────────────────────────────────────────────────────────

_US_ETFS: list[dict] = [
    # 宽基指数
    {"代码": "SPY",  "名称": "标普500ETF(SPDR)",       "类别": "宽基指数"},
    {"代码": "VOO",  "名称": "标普500ETF(先锋)",        "类别": "宽基指数"},
    {"代码": "QQQ",  "名称": "纳斯达克100ETF",          "类别": "宽基指数"},
    {"代码": "IWM",  "名称": "罗素2000ETF",             "类别": "宽基指数"},
    {"代码": "DIA",  "名称": "道琼斯ETF",               "类别": "宽基指数"},
    {"代码": "VTI",  "名称": "全市场ETF(先锋)",         "类别": "宽基指数"},
    {"代码": "MDY",  "名称": "标普400中盘ETF",          "类别": "宽基指数"},
    # 行业
    {"代码": "XLK",  "名称": "科技板块ETF",             "类别": "行业ETF"},
    {"代码": "SOXX", "名称": "半导体ETF(iShares)",      "类别": "行业ETF"},
    {"代码": "XLF",  "名称": "金融板块ETF",             "类别": "行业ETF"},
    {"代码": "XLE",  "名称": "能源板块ETF",             "类别": "行业ETF"},
    {"代码": "XLV",  "名称": "医疗板块ETF",             "类别": "行业ETF"},
    {"代码": "XLI",  "名称": "工业板块ETF",             "类别": "行业ETF"},
    {"代码": "XLP",  "名称": "必需消费ETF",             "类别": "行业ETF"},
    {"代码": "XLY",  "名称": "可选消费ETF",             "类别": "行业ETF"},
    {"代码": "XLRE", "名称": "房地产板块ETF",           "类别": "行业ETF"},
    {"代码": "ARKK", "名称": "ARK创新ETF",              "类别": "行业ETF"},
    # 加密货币
    {"代码": "IBIT", "名称": "比特币ETF(贝莱德)",       "类别": "加密货币"},
    {"代码": "FBTC", "名称": "比特币ETF(富达)",         "类别": "加密货币"},
    {"代码": "GBTC", "名称": "比特币信托(灰度)",        "类别": "加密货币"},
    {"代码": "ETHA", "名称": "以太坊ETF(贝莱德)",       "类别": "加密货币"},
    # 大宗商品
    {"代码": "GLD",  "名称": "黄金ETF(SPDR)",           "类别": "大宗商品"},
    {"代码": "IAU",  "名称": "黄金ETF(iShares)",        "类别": "大宗商品"},
    {"代码": "GDX",  "名称": "黄金矿业ETF(VanEck)",     "类别": "大宗商品"},
    {"代码": "SLV",  "名称": "白银ETF(iShares)",        "类别": "大宗商品"},
    {"代码": "USO",  "名称": "原油ETF",                 "类别": "大宗商品"},
    # 债券
    {"代码": "TLT",  "名称": "20年期美债ETF",           "类别": "债券ETF"},
    {"代码": "AGG",  "名称": "综合债券ETF(iShares)",    "类别": "债券ETF"},
    {"代码": "HYG",  "名称": "高收益债ETF",             "类别": "债券ETF"},
    {"代码": "LQD",  "名称": "投资级债ETF",             "类别": "债券ETF"},
    # 国际/新兴市场
    {"代码": "EEM",  "名称": "新兴市场ETF(iShares)",    "类别": "国际ETF"},
    {"代码": "EFA",  "名称": "发达市场ETF(iShares)",    "类别": "国际ETF"},
    {"代码": "FXI",  "名称": "中国大盘ETF",             "类别": "国际ETF"},
    {"代码": "KWEB", "名称": "中国互联网ETF(KraneShares)","类别": "国际ETF"},
    {"代码": "EWJ",  "名称": "日本ETF(iShares)",        "类别": "国际ETF"},
    {"代码": "EWZ",  "名称": "巴西ETF(iShares)",        "类别": "国际ETF"},
    # 杠杆/反向
    {"代码": "TQQQ", "名称": "纳指3倍做多ETF",          "类别": "杠杆ETF"},
    {"代码": "SOXL", "名称": "半导体3倍做多ETF",        "类别": "杠杆ETF"},
    {"代码": "UPRO", "名称": "标普3倍做多ETF",          "类别": "杠杆ETF"},
    {"代码": "SQQQ", "名称": "纳指3倍做空ETF",          "类别": "杠杆ETF"},
    {"代码": "SPXS", "名称": "标普3倍做空ETF",          "类别": "杠杆ETF"},
]


# ── US stocks ─────────────────────────────────────────────────────────────────

# Major non-S&P-500 US-listed stocks (large ADRs + other large caps) to supplement the S&P 500 list
_US_SUPPLEMENT = [
    "TSM","ASML","NVO","SAP","SHEL","AZN","TM","SONY","NVS","RY","TD","SNY",
    "HDB","INFY","WIT","BIDU","PDD","BABA","JD","TCEHY","MELI","SE","GRAB",
    "NET","DDOG","SNOW","CRWD","ZS","PANW","PLTR","ARM","SMCI","MRVL","AMAT",
    "LRCX","KLAC","MCHP","ADI","TXN","QCOM","INTC","MU","STX","WDC","SWKS",
]

def _fetch_us_ticker_list() -> list[str]:
    """Get S&P 500 tickers from Wikipedia + supplement. Cached 7 days."""
    cache_key = "us_ticker_list_v1"
    cached = _get(cache_key, ttl=86400 * 7)
    if cached:
        return cached

    tickers: set[str] = set(_US_SUPPLEMENT)
    try:
        from yfinance.data import YfData
        import pandas as pd, io
        yd = YfData()
        r  = yd.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=15)
        sp500_tickers = pd.read_html(io.StringIO(r.text))[0]["Symbol"] \
                          .str.replace(".", "-", regex=False).tolist()
        tickers.update(sp500_tickers)
    except Exception as exc:
        import sys; print(f"[us_tickers] wiki S&P500: {exc}", file=sys.stderr)

    result = sorted(tickers)
    if len(result) > 50:
        _set(cache_key, result)
    return result


def _yf_quote_batch(symbols: list[str]) -> dict:
    """Batch-fetch quote data (price, market cap, name, sector) via Yahoo Finance V7 API."""
    from yfinance.data import YfData
    yd      = YfData()
    results = {}
    BATCH   = 100
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            r = yd.get(
                "https://query2.finance.yahoo.com/v7/finance/quote",
                params={
                    "symbols": ",".join(batch),
                    "fields":  "symbol,shortName,marketCap,regularMarketPrice,"
                               "regularMarketChangePercent,sectorKey,trailingPE",
                },
                timeout=15,
            )
            for q in (r.json().get("quoteResponse") or {}).get("result") or []:
                sym = q.get("symbol")
                if sym:
                    results[sym] = q
        except Exception as exc:
            import sys; print(f"[yf_quote_batch] batch {i//BATCH}: {exc}", file=sys.stderr)
    return results


_US_CN_NAMES: dict[str, str] = {
    "NVDA": "英伟达", "AAPL": "苹果", "GOOGL": "谷歌(A)", "GOOG": "谷歌(C)",
    "MSFT": "微软", "AMZN": "亚马逊", "TSM": "台积电", "AVGO": "博通",
    "META": "Meta", "TSLA": "特斯拉", "BRK-B": "伯克希尔B", "LLY": "礼来",
    "MU": "美光科技", "WMT": "沃尔玛", "JPM": "摩根大通", "AMD": "超微半导体",
    "ASML": "阿斯麦", "XOM": "埃克森美孚", "V": "Visa", "ORCL": "甲骨文",
    "JNJ": "强生", "TCEHY": "腾讯控股", "INTC": "英特尔", "CSCO": "思科",
    "MA": "万事达", "COST": "好市多", "CAT": "卡特彼勒", "ABBV": "艾伯维",
    "BAC": "美国银行", "LRCX": "泛林半导体", "CVX": "雪佛龙", "ARM": "ARM控股",
    "UNH": "联合健康", "AMAT": "应用材料", "NFLX": "奈飞", "GE": "通用电气航空",
    "KO": "可口可乐", "PG": "宝洁", "MS": "摩根士丹利", "PLTR": "Palantir",
    "HD": "家得宝", "GS": "高盛", "MRK": "默克", "BABA": "阿里巴巴",
    "AZN": "阿斯利康", "NVS": "诺华", "PM": "菲利普莫里斯", "RY": "加拿大皇家银行",
    "IBM": "IBM", "TXN": "德州仪器", "DELL": "戴尔", "KLAC": "科磊",
    "GEV": "GE新能源", "WFC": "富国银行", "RTX": "雷神技术", "SHEL": "壳牌",
    "LIN": "林德", "SNDK": "闪迪", "TM": "丰田汽车", "QCOM": "高通",
    "C": "花旗集团", "PANW": "Palo Alto", "SAP": "SAP", "AXP": "美国运通",
    "MCD": "麦当劳", "ADI": "亚德诺半导体", "ANET": "Arista网络", "PEP": "百事可乐",
    "TMUS": "T-Mobile", "STX": "希捷", "NVO": "诺和诺德", "VZ": "威瑞森",
    "AMGN": "安进", "APP": "AppLovin", "TD": "道明银行", "NEE": "下一代能源",
    "TJX": "TJX公司", "WDC": "西部数据", "TMO": "赛默飞", "DIS": "迪士尼",
    "CRWD": "CrowdStrike", "APH": "安费诺", "BA": "波音", "BLK": "贝莱德",
    "UNP": "联合太平洋", "GILD": "吉利德", "ABT": "雅培", "T": "AT&T",
    "DE": "迪尔", "SCHW": "嘉信理财", "ETN": "伊顿", "GLW": "康宁",
    "CRM": "Salesforce", "ISRG": "直觉外科", "PFE": "辉瑞", "WELL": "韦尔塔",
    "UBER": "优步", "IBKR": "盈透证券", "COP": "康菲石油", "BX": "黑石",
    "HON": "霍尼韦尔", "PLD": "普洛斯", "DHR": "丹纳赫", "SONY": "索尼",
    "BKNG": "Booking", "CB": "丘博", "SPGI": "标普全球", "CVS": "CVS健康",
    "PDD": "拼多多", "LMT": "洛克希德马丁", "MO": "奥驰亚", "HDB": "HDFC银行",
    "PGR": "前进保险", "LOW": "劳氏", "SYK": "史赛克", "BMY": "百时美施贵宝",
    "NOW": "ServiceNow", "VRT": "Vertiv", "VRTX": "福泰制药", "COF": "第一资本",
    "PH": "派克汉尼汾", "ACN": "埃森哲", "SBUX": "星巴克", "SNY": "赛诺菲",
    "EQIX": "Equinix", "NEM": "纽蒙特", "FTNT": "飞塔信息", "MDT": "美敦力",
    "SO": "南方公司", "PWR": "Quanta服务", "CDNS": "铿腾电子", "MAR": "万豪",
    "ADBE": "Adobe", "TT": "特灵科技", "HWM": "豪梅特航空", "BNY": "纽约梅隆银行",
    "DUK": "杜克能源", "GD": "通用动力", "MCK": "麦克森", "CME": "芝商所",
    "ADP": "自动数据处理", "UPS": "联合包裹", "PNC": "PNC金融",
    "FCX": "自由港麦克莫兰", "CEG": "星座能源", "AMT": "美国铁塔",
    "ELV": "卓越健康", "CMI": "康明斯", "SNPS": "新思科技", "WM": "废物管理",
    "NET": "Cloudflare", "WMB": "威廉姆斯公司", "JCI": "江森自控",
    "MNST": "魔爪饮料", "CSX": "CSX", "KKR": "KKR", "USB": "美国合众银行",
    "CMCSA": "康卡斯特", "DDOG": "Datadog", "SNOW": "雪花", "HCA": "HCA医疗",
    "SLB": "斯伦贝谢", "MELI": "美客多", "INTU": "财捷", "MMM": "3M",
    "ICE": "洲际交易所", "SPG": "西蒙地产", "MRSH": "威达信", "MDLZ": "亿滋国际",
    "ABNB": "爱彼迎", "FDX": "联邦快递", "MCO": "穆迪", "HLT": "希尔顿",
    "EMR": "艾默生电气", "NOC": "诺斯罗普格鲁曼", "CI": "信诺",
    "MPC": "马拉松炼化", "VLO": "瓦莱罗能源", "SHW": "宣威",
    "RCL": "皇家加勒比", "ORLY": "奥莱利汽车", "NXPI": "恩智浦",
    "HOOD": "Robinhood", "GM": "通用汽车", "ROST": "罗斯百货",
    "APO": "阿波罗全球", "COHR": "Coherent", "PSX": "菲利普斯66",
    "EOG": "EOG资源", "CVNA": "卡瓦纳", "MPWR": "芯源系统",
    "ITW": "伊利诺伊工具", "ECL": "艺康", "BSX": "波士顿科学",
    "CTAS": "新思达", "CL": "高露洁棕榄", "KMI": "金德摩根",
    "NSC": "诺福克南方", "AEP": "美国电力", "CRH": "CRH", "AON": "怡安",
    "TDG": "TransDigm", "CIEN": "赛恩纳", "DASH": "DoorDash",
    "MSI": "摩托罗拉方案", "LITE": "Lumentum", "URI": "联合租赁",
    "DLR": "数字房地产", "REGN": "再生元", "WBD": "华纳兄弟探索",
    "HPE": "慧与科技", "FIX": "舒适系统", "RSG": "共和国服务",
    "TRV": "旅行者", "NKE": "耐克", "APD": "空气化工", "BKR": "贝克休斯",
    "TEL": "泰科电子", "PCAR": "PACCAR", "GWW": "固安捷",
    "TFC": "特鲁斯特金融", "AFL": "美国家庭人寿", "SRE": "森普拉",
    "F": "福特汽车", "D": "道明尼能源", "NUE": "纽柯钢铁",
    "LHX": "L3哈里斯", "ALL": "好事达", "O": "房地产收入",
    "TRGP": "Targa资源", "OXY": "西方石油", "KEYS": "是德科技",
    "TER": "泰瑞达", "CARR": "开利全球", "TGT": "塔吉特",
    "OKE": "ONEOK", "AJG": "盖洛格", "MET": "大都会人寿",
    "PSA": "公共存储", "FANG": "钻石能源", "FAST": "法斯纳尔",
    "COR": "Cencora", "SE": "Sea Limited", "DAL": "达美航空",
    "AME": "AMETEK", "CTVA": "科迪华", "DVN": "德文能源",
    "AZO": "汽车地带", "EA": "艺电", "ETR": "恩特基",
    "ODFL": "老多明尼货运", "INFY": "印孚瑟斯", "VST": "Vistra能源",
    "ROK": "罗克韦尔自动化", "EW": "爱德华兹生命科学", "NDAQ": "纳斯达克",
    "XEL": "Xcel能源", "ADSK": "欧特克", "EBAY": "eBay",
    "CAH": "卡地纳健康", "MCHP": "微芯科技", "FITB": "第五三银行",
    "EXC": "Exelon", "GRMN": "佳明", "ON": "安森美", "STT": "道富银行",
    "MSCI": "MSCI", "IDXX": "爱德士", "WAB": "西屋气刹", "HUM": "人力资源管理",
    "BDX": "宝迪", "YUM": "百胜餐饮", "KDP": "克里格胡椒博士",
    "ARES": "Ares管理", "BIDU": "百度", "DHI": "D.R.霍顿",
    "CCI": "皇冠城堡", "AMP": "安普理财", "XYZ": "Block",
    "COIN": "Coinbase", "AIG": "美国国际集团", "VTR": "万塔斯",
    "TTWO": "双城互动", "PEG": "公共服务企业", "KR": "克罗格",
    "AXON": "Axon", "ED": "合并爱迪生", "ADM": "阿彻丹尼尔斯",
    "JD": "京东", "TKO": "TKO集团", "STLD": "钢铁动力",
    "CBRE": "世邦魏理仕", "A": "安捷伦", "CCL": "嘉年华邮轮",
    "PCG": "太平洋电气", "CMG": "奇波雷", "HSY": "好时",
    "JBL": "捷普科技", "LYV": "Live Nation", "IRM": "铁山",
    "WEC": "WEC能源", "VMC": "火神材料", "SYY": "西斯科",
    "PYPL": "PayPal", "EME": "EMCOR集团", "PRU": "保德信金融",
    "HIG": "哈特福德", "PAYX": "Paychex", "WAT": "沃特世",
    "WDAY": "Workday", "MLM": "马丁玛丽埃塔", "UAL": "美联航",
    "KVUE": "科美", "SATS": "EchoStar", "EQT": "EQT",
    "ROP": "罗珀技术", "HBAN": "亨廷顿银行", "LVS": "拉斯维加斯金沙",
    "ZTS": "硕腾", "NTAP": "NetApp", "KMB": "金佰利",
    "HAL": "哈利伯顿", "MTB": "M&T银行", "EXR": "Extra Space Storage",
    "ACGL": "拱形资本", "NTRS": "北方信托", "CNC": "森泰尼",
    "IQV": "IQVIA", "DTE": "DTE能源", "AEE": "艾默伦", "EL": "雅诗兰黛",
}


def _py_initials(name: str) -> str:
    """Return lowercase pinyin initials string for a Chinese name, e.g. '英伟达' → 'ywd'."""
    try:
        from pypinyin import lazy_pinyin, Style
        return "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
    except Exception:
        return ""


def _fetch_us_stocks() -> list[dict]:
    """Fetch US large-cap stocks (≥ 300亿USD = $30B) via Yahoo Finance."""
    cache_key = "us_stocks_v3"
    cached = _get(cache_key, ttl=300)   # 5-minute TTL for real-time prices
    if cached:
        return cached

    tickers = _fetch_us_ticker_list()
    quotes  = _yf_quote_batch(tickers)

    MIN_MKT = 300.0   # 亿USD
    stocks: list[dict] = []
    for sym, q in quotes.items():
        mktcap = float(q.get("marketCap") or 0) / 1e8
        if mktcap < MIN_MKT:
            continue
        price  = round(float(q.get("regularMarketPrice") or 0), 2)
        chg    = round(float(q.get("regularMarketChangePercent") or 0), 2)
        name   = str(q.get("shortName") or sym)
        sector = str(q.get("sectorKey") or "")
        pe_raw = q.get("trailingPE")
        pe     = round(float(pe_raw), 1) if pe_raw and float(pe_raw) > 0 else 0.0
        stocks.append({
            "代码":          sym,
            "名称":          name,
            "市值亿":        round(mktcap, 1),
            "最新价":        price,
            "涨跌幅":        chg,
            "pe":            pe,
            "industry_board": sector or "美股",
            "_tc":           f"us{sym}",
        })

    stocks.sort(key=lambda x: -x["市值亿"])
    if stocks:
        _set(cache_key, stocks)
    return stocks


def _us_hfq_k(raw_hist) -> float:
    """Compute 后复权 K from a 10-year auto_adjust=False history DataFrame.
    K = cumulative_10yr_split_ratio × (raw_close[0] / adj_close[0])
    后复权_price(t) = adj_close(t) × K
    """
    try:
        split_k = 1.0
        for v in raw_hist.get("Stock Splits", []):
            if float(v) > 1:
                split_k *= float(v)
        c0 = float(raw_hist["Close"].iloc[0])
        a0 = float(raw_hist["Adj Close"].iloc[0])
        return split_k * c0 / a0 if a0 > 0 else split_k
    except Exception:
        return 1.0


def _us_hfq_k_cached(ticker: str) -> float:
    """Return 后复权 K for a US ticker, always derived from a 10-year window.
    Cached 7 days — splits/dividends are infrequent.
    Both _fetch_us_kline_em and _yf_ohlcv use this so short chart windows
    (6 m daily, 2 yr weekly) don't miss splits that happened before the window.
    """
    ck = f"us_hfq_k_{ticker}"
    v  = _get(ck, ttl=86400 * 7)
    if v is not None:
        return float(v)
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(period="10y", auto_adjust=False, actions=True)
        K   = _us_hfq_k(raw) if not raw.empty else 1.0
    except Exception:
        K = 1.0
    _set(ck, K)
    return K


def _fetch_us_kline_em(ticker: str, years: int = 10) -> tuple[list[str], list[float]]:
    """Fetch 后复权 daily kline for a US stock via yfinance."""
    cache_key = f"us_kline_v3_{ticker}"
    cached = _get(cache_key, ttl=86400)
    if cached:
        return cached.get("dates", []), cached.get("closes", [])
    try:
        import yfinance as yf
        period = f"{years}y"
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=True)
        if raw.empty:
            return [], []
        K = _us_hfq_k(raw)
        # Backfill the shared K cache so _yf_ohlcv doesn't need a second 10yr fetch
        _set(f"us_hfq_k_{ticker}", K)
        import math as _math
        dates, closes = [], []
        for d, ac in zip(raw.index, raw["Adj Close"]):
            ac_f = float(ac)
            if not _math.isnan(ac_f):
                dates.append(str(d)[:10])
                closes.append(round(ac_f * K, 4))
        if dates:
            _set(cache_key, {"dates": dates, "closes": closes})
        return dates, closes
    except Exception as exc:
        import sys; print(f"[us_kline] {ticker}: {exc}", file=sys.stderr)
        return [], []


# ── trend trading scan ────────────────────────────────────────────────────────

def _fetch_recent_highs(tc_code: str) -> tuple[float, float, float, float, float | None]:
    """Return (max_close_50d, max_close_50d, intraday_price, atr_50d).

    intraday_price = yesterday's hfq close scaled by today's real-time chg%,
    so it stays on the same hfq basis as max_close_50d.
    Kline data (max_c50, last_hfq, atr) cached 2 h; real-time price fetched fresh.
    Returns (0.0, 0.0, 0.0, 0.0) on failure.
    """
    # ── slow-changing kline data: cached 2 h ─────────────────────────────────
    cache_key = f"rec_highs_v8_{tc_code}"
    cached = _get(cache_key, ttl=43200)
    rows = []   # needed for prev_vol below; stays [] on cache-hit path
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
            dates  = [row[0] for row in rows if len(row) > 2 and row[2]]
            closes = [float(row[2]) for row in rows if len(row) > 2 and row[2]]
            if len(closes) >= 10:
                today_str = today.strftime("%Y-%m-%d")
                if dates and dates[-1] >= today_str and len(closes) >= 2:
                    # kline 包含今日未完成 bar：昨收作为基准，max 只用已完成历史收盘
                    last_hfq   = closes[-2]
                    hist_close = closes[:-1]
                else:
                    last_hfq   = closes[-1]
                    hist_close = closes
                max_c50 = max(hist_close[-50:]) if len(hist_close) >= 50 else max(hist_close)
                rets = [abs(closes[i] / closes[i-1] - 1) * 100
                        for i in range(1, len(closes))]
                atr  = round(sum(rets[-50:]) / min(len(rets), 50), 2)
                _set(cache_key, [round(max_c50, 4), round(last_hfq, 4), atr])
        except Exception:
            pass

    if last_hfq == 0.0:
        return (0.0, 0.0, 0.0, 0.0, None)

    # ── real-time intraday price: apply today's chg% to hfq last close ───────
    # 与日K逻辑一致：若 qt.gtimg 成交量 == kline 最后一条成交量，说明今天尚未开盘，
    # qt.gtimg 返回的是昨天数据，不能再乘一次昨天涨跌幅，直接用 last_hfq。
    cur_price = last_hfq
    chg_pct   = None
    prev_vol  = float(rows[-1][5]) if rows and len(rows[-1]) > 5 else -1
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={tc_code}", timeout=5)
        for ln in r.text.strip().split(";\n"):
            if not ln.strip():
                continue
            flds = ln.split('"')[1].split("~")
            if len(flds) > 32 and flds[32]:
                qt_vol = float(flds[6] or 0)
                chg    = float(flds[32])
                if qt_vol != prev_vol and qt_vol > 0:
                    # 今天有新成交 → 用实时涨跌幅估算后复权当前价
                    chg_pct   = round(chg, 2)
                    cur_price = round(last_hfq * (1 + chg / 100), 4)
                else:
                    # 尚未开盘或成交量未变 → cur_price = 昨收，涨跌幅 = 0
                    chg_pct   = 0.0
                    cur_price = last_hfq
            break
    except Exception:
        pass

    # max_high_50d_b = last_hfq (stable close for drawdown, not intraday cur_price)
    # Using intraday cur_price distorts drawdown when today is a big move
    return (round(max_c50, 4), round(last_hfq, 4), cur_price, atr, chg_pct)


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
    22-week cache is not invalidated.  Cached 12 h.
    """
    cache_key = f"weekly_long_v1_{tc_code}"
    cached = _get(cache_key, ttl=43200)
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
    if not result and not tc_code.startswith("hk"):
        result = _fetch_weekly_closes_em(tc_code, n)
    if result:
        _set(cache_key, result)
    return result


def _fetch_weekly_closes_em(tc_code: str, n: int) -> list[float]:
    """Eastmoney fallback for A-share weekly hfq closes (push2his klt=102)."""
    end = _weekly_end_date()
    beg_date = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=(n + 6) * 7)).strftime("%Y%m%d")
    end_date = end.replace("-", "")
    mkt = "1" if tc_code.startswith("sh") else "0"
    code = tc_code[2:]
    secid = f"{mkt}.{code}"
    try:
        r = _em_sess.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": secid, "fields1": "f1,f2,f3,f4,f5",
                "fields2": "f51,f52,f53,f54,f55,f56",
                "klt": "102", "fqt": "1",
                "beg": beg_date, "end": end_date, "lmt": str(n + 10),
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
            if parts[0].replace("-", "") > end_date:
                continue
            try:
                closes.append(float(parts[2]))
            except (ValueError, IndexError):
                pass
        return closes[-n:]
    except Exception:
        return []


def _fetch_weekly_closes(tc_code: str, n: int = 12) -> list[float]:
    """Return last n weekly hfq-adjusted closing prices, oldest first.

    Primary: Tencent fqkline 'week' endpoint.
    Fallback: Eastmoney push2his klt=102 (when Tencent is rate-limited).
    Cached 12 h.
    """
    cache_key = f"weekly_v2_{tc_code}"
    cached = _get(cache_key, ttl=43200)
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
    # Eastmoney fallback for A-shares/ETFs when Tencent fails (e.g. rate-limit from non-CN IPs)
    if not result and not tc_code.startswith("hk"):
        result = _fetch_weekly_closes_em(tc_code, n)
    if result:
        _set(cache_key, result)
    return result


# Trend scan state (independent of the main screener scan)
_trend_scan: dict = {
    "running": False, "done": False, "idx": 0, "total": 0, "results": []
}
_trend_lock = threading.Lock()


def _fetch_us_weekly_closes_scan(ticker: str, n: int = 22) -> list[float]:
    """Return last n weekly 后复权 closes for US trend scan. Cached 2 h."""
    ck = f"us_wkly_scan_{ticker}"
    cached = _get(ck, ttl=7200)
    if cached:
        return cached[-n:]
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(period="2y", interval="1wk",
                                         auto_adjust=False, actions=True)
        if raw.empty:
            return []
        K = _us_hfq_k_cached(ticker)
        closes = [round(float(c) * K, 4) for c in raw["Adj Close"]]
        _set(ck, closes)
        return closes[-n:]
    except Exception:
        return []


def _fetch_us_recent_highs_scan(ticker: str):
    """Return (max_h50, max_h50, last_close, atr_50d, chg_live) for a US ticker."""
    ck = f"us_rh_scan_{ticker}"
    cached = _get(ck, ttl=7200)
    if cached:
        return tuple(cached)
    try:
        import yfinance as yf
        t   = yf.Ticker(ticker)
        K   = _us_hfq_k_cached(ticker)
        raw = t.history(period="3mo", interval="1d", auto_adjust=False, actions=False)
        if raw.empty or len(raw) < 5:
            return 0.0, 0.0, 0.0, 0.0, None
        closes   = [float(c) * K for c in raw["Adj Close"]]
        max_h50  = max(closes[-50:]) if len(closes) >= 50 else max(closes)
        last_c   = closes[-1]
        rets     = [abs(closes[i] / closes[i-1] - 1) * 100 for i in range(1, len(closes))]
        atr      = round(sum(rets[-50:]) / min(len(rets), 50), 2) if rets else 0.0
        chg_live = None
        try:
            fi       = t.fast_info
            chg_live = round(float(fi.get("regularMarketChangePercent") or 0), 2)
        except Exception:
            pass
        res = [round(max_h50, 4), round(max_h50, 4), round(last_c, 4), atr, chg_live]
        _set(ck, res)
        return tuple(res)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, None


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
        is_us  = bool(stock.get("is_us"))
        try:
            n_weeks = 7 if is_etf else 10
            if is_us:
                closes_all = _fetch_us_weekly_closes_scan(code, n=22)
            else:
                closes_all = _fetch_weekly_closes(tc, n=22)
            if len(closes_all) < 5:
                return None
            closes = closes_all[-n_weeks:]
            n_pts  = len(closes)
            x = np.arange(n_pts, dtype=float)
            y = np.log(np.array(closes, dtype=float))
            slope, b_intercept, r2 = _ols(x, y)

            y_pred    = slope * x + b_intercept
            sigma_res = round(float(np.std(y - y_pred)) * 100, 3)

            p0   = closes[0]
            norm = [round(p / p0, 4) for p in closes]

            if is_etf:
                industry = stock.get("industry_board", "ETF")
            elif is_us:
                industry = stock.get("industry_board", "") or "美股"
            elif is_hk:
                industry = stock.get("industry_board", "") or "港股其他"
            else:
                industry = _fetch_industry_board(code)

            if is_us:
                max_h50, max_h50b, last_cls, atr_50d, chg_live = _fetch_us_recent_highs_scan(code)
            else:
                max_h50, max_h50b, last_cls, atr_50d, chg_live = _fetch_recent_highs(tc)

            ret10w = None
            if len(closes_all) >= 10:
                c0, c1 = closes_all[-10], closes_all[-1]
                if c0 > 0:
                    ret10w = round((c1 / c0 - 1) * 100, 1)

            ret20w = None
            if len(closes_all) >= 20:
                c0, c1 = closes_all[-20], closes_all[-1]
                if c0 > 0:
                    ret20w = round((c1 / c0 - 1) * 100, 1)

            chg_today = chg_live if chg_live is not None else stock.get("涨跌幅")

            return {
                "代码":          code,
                "名称":          stock["名称"],
                "市值亿":        stock["市值亿"],
                "pe":            stock.get("pe", 0.0),
                "is_etf":        is_etf,
                "is_hk":         is_hk,
                "is_us":         is_us,
                "_tc":           tc,
                "trend_slope":   round(slope * 100, 2),
                "trend_r2":      round(r2 * 100, 1),
                "sigma_res":     sigma_res,
                "trend_closes":  norm,
                "industry_board": industry,
                "max_high_50d":  max_h50,
                "max_high_50d_b": max_h50b,
                "last_close":    last_cls,
                "atr_50d":       atr_50d,
                "ret10w":        ret10w,
                "ret20w":        ret20w,
                "涨跌幅":        chg_today,
            }
        except Exception:
            return None
        finally:
            with _trend_lock:
                _trend_scan["idx"] += 1

    # US: more workers (yfinance handles concurrency); A/HK: 4 to avoid Tencent rate-limit
    is_us_scan = any(s.get("is_us") for s in stock_list[:3])
    n_workers  = 8 if is_us_scan else 4
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
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


@app.route("/api/etf_quotes")
def etf_quotes():
    """返回 _MAJOR_ETFS 列表加实时行情（价格、涨跌幅、成交量）。"""
    tc_list = [e["_tc"] for e in _MAJOR_ETFS]
    quotes: dict = {}
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={','.join(tc_list)}", timeout=8)
        r.raise_for_status()
        for line in r.text.strip().split(";\n"):
            if '="' not in line: continue
            var   = line.split("=")[0].strip()
            tc    = var[2:] if var.startswith("v_") else var
            inner = line.split('="', 1)[1].rstrip('";')
            flds  = inner.split("~")
            if len(flds) >= 10:
                price = float(flds[3] or 0)
                prev  = float(flds[4] or 0)
                vol   = float(flds[6] or 0)   # 手
                amt   = float(flds[37] or 0) if len(flds) > 37 else 0  # 万元
                chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
                quotes[tc] = {"price": price, "chg_pct": chg, "volume": vol, "amount": amt}
    except Exception:
        pass

    result = []
    for e in _MAJOR_ETFS:
        q = quotes.get(e["_tc"], {})
        name = e["名称"]
        result.append({
            "代码":   e["代码"],
            "名称":   name,
            "类别":   e["industry_board"],
            "_tc":    e["_tc"],
            "价格":   q.get("price", 0),
            "涨跌幅": q.get("chg_pct", 0),
            "成交量": q.get("volume", 0),
            "成交额": q.get("amount", 0),
            "py":     _py_initials(name),
        })
    return jsonify({"success": True, "data": result})


@app.route("/api/us_etf_quotes")
def us_etf_quotes():
    """Return US ETF quotes via yfinance batch API."""
    tickers = [e["代码"] for e in _US_ETFS]
    quotes  = _yf_quote_batch(tickers)
    result  = []
    for e in _US_ETFS:
        q     = quotes.get(e["代码"], {})
        price = float(q.get("regularMarketPrice") or 0)
        chg   = float(q.get("regularMarketChangePercent") or 0)
        result.append({
            "代码":   e["代码"],
            "名称":   e["名称"],
            "类别":   e["类别"],
            "价格":   round(price, 2),
            "涨跌幅": round(chg, 2),
        })
    return jsonify({"success": True, "data": result})


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
        import math as _math
        from collections import defaultdict
        all_stocks = _fetch_a_stock_list()
        if not all_stocks:
            return jsonify({"success": False, "error": "股票列表为空，请稍后重试"}), 503

        # 按申万二级行业分桶，每桶取市值 top 10%（向下取整）
        buckets: dict[str, list] = defaultdict(list)
        for s in all_stocks:
            buckets[s.get("行业") or "其他"].append(s)

        stocks = []
        for group in buckets.values():
            group.sort(key=lambda x: x["市值亿"], reverse=True)
            n = _math.floor(len(group) * 0.1)
            stocks.extend(group[:n])

        stocks.sort(key=lambda x: x["市值亿"], reverse=True)
        for s in stocks:
            s["py"] = _py_initials(s.get("名称", ""))
        payload = {"success": True, "data": stocks, "total": len(stocks)}
        _set("stock_list", payload)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/hk_stocks")
def get_hk_stocks():
    """返回市值 ≥ 500亿港元的港股列表，附实时行情（价格、涨跌幅）。"""
    MIN_MKT = 500.0  # 亿港元
    stocks = [s for s in _fetch_hk_stocks() if s["市值亿"] >= MIN_MKT]
    stocks.sort(key=lambda s: -s["市值亿"])

    # 批量拉腾讯实时行情（价格 + 涨跌幅）
    tc_list = [s["_tc"] for s in stocks]
    quotes: dict = {}
    BATCH = 200
    for i in range(0, len(tc_list), BATCH):
        try:
            r = _sess.get(f"https://qt.gtimg.cn/q={','.join(tc_list[i:i+BATCH])}", timeout=8)
            for line in r.text.strip().split(";\n"):
                if '="' not in line: continue
                var   = line.split("=")[0].strip()
                tc    = var[2:] if var.startswith("v_") else var
                inner = line.split('="', 1)[1].rstrip('";')
                flds  = inner.split("~")
                if len(flds) >= 5:
                    price = float(flds[3] or 0)
                    prev  = float(flds[4] or 0)
                    chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
                    quotes[tc] = {"最新价": price, "涨跌幅": chg}
        except Exception:
            pass

    for s in stocks:
        q = quotes.get(s["_tc"], {})
        s["最新价"] = q.get("最新价", 0)
        s["涨跌幅"] = q.get("涨跌幅", s.get("涨跌幅", 0))
        s["py"] = _py_initials(s.get("名称", ""))

    return jsonify({"success": True, "data": stocks, "total": len(stocks)})


@app.route("/api/us_stocks")
def get_us_stocks():
    """返回市值 ≥ 300亿美元的美股列表（数据全部来自 Yahoo Finance）。"""
    stocks = _fetch_us_stocks()
    if not stocks:
        return jsonify({"success": False, "error": "获取美股数据失败"}), 503
    for s in stocks:
        cn = _US_CN_NAMES.get(s.get("代码", ""))
        if cn:
            s["名称"] = cn
        s["py"] = _py_initials(s.get("名称", ""))
    return jsonify({"success": True, "data": stocks, "total": len(stocks)})


@app.route("/api/stock/<code>")
def get_stock_data(code: str):
    try:
        tc_code = _resolve_tc(code)
        p = _compute_regression_payload(tc_code, code)
        if not p:
            return jsonify({"success": False, "error": "无历史数据"}), 404
        # PE history: A-shares via BaoStock, HK/US via yfinance annual EPS
        if tc_code.startswith("us"):
            pe = _compute_pe_payload_us(code)
        elif tc_code.startswith("hk"):
            pe = _compute_pe_payload_hk(code)
        else:
            pe = _compute_pe_payload(tc_code, code)
        if pe:
            p = {**p, **pe}
        p["currency"] = "USD" if tc_code.startswith("us") else "HKD" if tc_code.startswith("hk") else "CNY"
        # Analyst consensus EPS forecasts
        # US: yfinance (USD), HK: yfinance (converted to HKD), A: Eastmoney
        if tc_code.startswith("us"):
            consensus = _fetch_us_consensus_yf(code)
        elif tc_code.startswith("hk"):
            consensus = _fetch_hk_consensus_yf(code)
        else:
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
    if code.startswith(("sz", "sh", "hk", "us")):
        return code
    if len(code) <= 5 and code.isdigit():
        return "hk" + code.zfill(5)
    # US stock ticker: letters only, or letters with one hyphen/dot suffix (BRK-B, BRK.B)
    import re as _re
    if _re.match(r'^[A-Z]{1,6}([.\-][A-Z]{1,2})?$', code):
        return f"us{code}"
    # ETF 列表优先（含正确 sh/sz 前缀）
    for e in _MAJOR_ETFS:
        if e["代码"] == code:
            return e["_tc"]
    # A 股列表
    cl = _get("stock_list", ttl=86400)
    if cl:
        for s in cl["data"]:
            if s["代码"] == code:
                return s["_tc"]
    # 兜底：5/6 开头 → 上交所；其余 → 深交所
    return ("sh" if code.startswith(("6", "5")) else "sz") + code


# ── Watchlist ─────────────────────────────────────────────────────────────────
_WATCHLIST_KEY  = "watchlist_v1"
_WATCHLIST_MAX  = 20
_wl_mem: list[dict] = []
_wl_lock = threading.Lock()
_WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".watchlist.json")


def _wl_load() -> list[dict]:
    if _redis:
        try:
            raw = _redis.get(_WATCHLIST_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    # 本地文件兜底
    try:
        with open(_WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    with _wl_lock:
        return list(_wl_mem)


def _wl_save(items: list[dict]):
    global _wl_mem
    with _wl_lock:
        _wl_mem = list(items)
    if _redis:
        try:
            _redis.setex(_WATCHLIST_KEY, 86400 * 365, json.dumps(items))
            return
        except Exception:
            pass
    # 本地文件兜底
    try:
        with open(_WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _wl_resolve(raw: str) -> dict | None:
    """Resolve raw user input to {code, tc_code, name}. Supports A股/港股."""
    s = raw.strip()
    if not s:
        return None
    if s.lower().startswith(("sz", "sh", "hk")):
        tc = s.lower()
        pure = tc[2:]
    elif len(s) == 6 and s.isdigit():
        tc = ("sh" if s.startswith(("6", "5")) else "sz") + s
        pure = s
    elif s.isdigit():
        pure = s.zfill(5)
        tc = "hk" + pure
    else:
        return None
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={tc}", timeout=6)
        r.raise_for_status()
        for line in r.text.strip().split(";\n"):
            if '="' not in line:
                continue
            inner = line.split('="', 1)[1].rstrip('";')
            fields = inner.split("~")
            if len(fields) >= 4:
                name  = fields[1].strip()
                code  = fields[2].strip() or pure
                price = float(fields[3] or 0)
                if name and price > 0:
                    return {"code": code, "tc_code": tc, "name": name}
    except Exception:
        pass
    return None


@app.route("/api/watchlist")
def watchlist_get():
    items = _wl_load()
    if not items:
        return jsonify({"success": True, "stocks": []})
    tc_list = [it["tc_code"] for it in items]
    quotes: dict[str, dict] = {}
    try:
        r = _sess.get(f"https://qt.gtimg.cn/q={','.join(tc_list)}", timeout=6)
        r.raise_for_status()
        for line in r.text.strip().split(";\n"):
            if '="' not in line:
                continue
            var = line.split("=")[0].strip()      # "v_sz000001"
            tc  = var[2:] if var.startswith("v_") else var
            inner = line.split('="', 1)[1].rstrip('";')
            fields = inner.split("~")
            if len(fields) >= 5:
                price = float(fields[3] or 0)
                prev  = float(fields[4] or 0)
                if price > 0 and prev > 0:
                    chg_pct = round((price - prev) / prev * 100, 2)
                    chg_amt = round(price - prev, 3)
                    quotes[tc] = {"price": price, "chg_pct": chg_pct, "chg_amt": chg_amt}
    except Exception:
        pass
    result = [{**it, **quotes.get(it["tc_code"], {})} for it in items]
    return jsonify({"success": True, "stocks": result})


@app.route("/api/watchlist/add", methods=["POST"])
def watchlist_add():
    body = request.get_json(silent=True) or {}
    raw  = str(body.get("code", "")).strip()
    if not raw:
        return jsonify({"error": "请输入代码"}), 400
    resolved = _wl_resolve(raw)
    if not resolved:
        return jsonify({"error": f"未找到股票：{raw}"}), 404
    items = _wl_load()
    if any(it["tc_code"] == resolved["tc_code"] for it in items):
        return jsonify({"error": "已在自选股中"}), 409
    if len(items) >= _WATCHLIST_MAX:
        return jsonify({"error": f"自选股最多 {_WATCHLIST_MAX} 只"}), 400
    items.append(resolved)
    _wl_save(items)
    return jsonify({"success": True, "stock": resolved})


@app.route("/api/watchlist/remove/<tc_code>", methods=["DELETE"])
def watchlist_remove(tc_code: str):
    items = _wl_load()
    new = [it for it in items if it["tc_code"] != tc_code]
    if len(new) == len(items):
        return jsonify({"error": "未找到"}), 404
    _wl_save(new)
    return jsonify({"success": True})


def _yf_ohlcv(ticker: str, start: str, end: str, interval: str) -> list[dict]:
    """Fetch 后复权 OHLCV bars from yfinance."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(start=start, end=end,
                                        interval=interval, auto_adjust=False, actions=True)
        if raw.empty:
            return []
        # Use 10yr-based K so out-of-window splits (e.g. GOOGL 20:1 in 2022)
        # are captured even when the chart window is only 6 months or 2 years.
        import math
        K = _us_hfq_k_cached(ticker)
        rows = []
        for ts, row in raw.iterrows():
            c   = float(row["Close"])
            ac  = float(row["Adj Close"])
            # Yahoo Finance sometimes returns NaN for Adj Close on the latest
            # bar; fall back to actual Close so the bar isn't skipped/broken.
            if math.isnan(ac):
                ac = c
            if math.isnan(c) or c <= 0:
                continue
            # per-bar dividend ratio to keep OHLC consistent with 后复权 close
            dr  = ac / c if c > 0 else 1.0
            rows.append({
                "date":   str(ts)[:10],
                "open":   round(float(row["Open"])  * dr * K, 4),
                "close":  round(ac * K,                       4),
                "high":   round(float(row["High"])  * dr * K, 4),
                "low":    round(float(row["Low"])   * dr * K, 4),
                "volume": float(row["Volume"]),
            })
        return rows
    except Exception as exc:
        import sys; print(f"[yf_ohlcv] {ticker} {interval}: {exc}", file=sys.stderr)
        return []


@app.route("/api/stock/kline/<code>")
def get_stock_kline(code: str):
    """Return hfq OHLC for daily (6 months) and weekly (2 years).
    Optional ?before=YYYY-MM-DD to load earlier historical data (load-more)."""
    tc = _resolve_tc(code)
    today = datetime.now()

    # before 参数：往前加载更多历史
    before_str = request.args.get("before", "")
    if before_str:
        try:
            end_dt = datetime.strptime(before_str, "%Y-%m-%d") - timedelta(days=1)
        except ValueError:
            end_dt = today
        end = end_dt.strftime("%Y-%m-%d")
        start_d = (end_dt - timedelta(days=300)).strftime("%Y-%m-%d")
        start_w = (end_dt - timedelta(days=900)).strftime("%Y-%m-%d")
    else:
        end     = today.strftime("%Y-%m-%d")
        start_d = (today - timedelta(days=200)).strftime("%Y-%m-%d")
        start_w = (today - timedelta(days=730)).strftime("%Y-%m-%d")

    # ── US stocks: use yfinance ───────────────────────────────────────────────
    if tc.startswith("us"):
        end_yf = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        daily  = _yf_ohlcv(code, start_d, end_yf, "1d")
        weekly = _yf_ohlcv(code, start_w, end_yf, "1wk")
        return jsonify({"success": True, "daily": daily, "weekly": weekly})

    def _fetch(freq, start, count):
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

    daily  = _fetch("day",  start_d, 200)
    weekly = _fetch("week", start_w, 130)

    # 盘中补今日实时 K 线（仅非 before 模式）
    today_str = today.strftime("%Y-%m-%d")
    if not before_str and daily and daily[-1]["date"] != today_str:
        try:
            r = _sess.get(f"https://qt.gtimg.cn/q={tc}", timeout=5)
            r.raise_for_status()
            for line in r.text.strip().split(";\n"):
                if '="' not in line:
                    continue
                inner = line.split('="', 1)[1].rstrip('";')
                f = inner.split("~")
                if len(f) >= 35:
                    o = float(f[5] or 0)
                    c = float(f[3] or 0)
                    h = float(f[33] or 0)
                    l = float(f[34] or 0)
                    v = float(f[6] or 0)
                    prev_vol = daily[-1]["volume"]
                    # 成交量与上一交易日相同 → 接口返回的是缓存旧数据（未开盘/休市）
                    if c > 0 and o > 0 and v != prev_vol:
                        # qt.gtimg 返回实际（未复权）价格，需换算成后复权价格。
                        # 系数 = 昨日 hfq 收盘 / 昨日实际收盘（field[4]）。
                        # 除权当天 field[4] 为除权前实际收盘价，hfq 已含历史复权，
                        # 两者之比即为当前累计复权系数，今日 OHLC 乘以该系数即可。
                        prev_actual = float(f[4] or 0)
                        prev_hfq    = daily[-1]["close"]
                        scale = (prev_hfq / prev_actual) if prev_actual > 0 else 1.0
                        daily.append({
                            "date": today_str,
                            "open":  round(o * scale, 4),
                            "close": round(c * scale, 4),
                            "high":  round(h * scale, 4),
                            "low":   round(l * scale, 4),
                            "volume": v, "intraday": True,
                        })
                break
        except Exception:
            pass

    # 补/更新本周实时周 K 线
    # 不论周线接口是否已有本周数据（可能只到昨天），都用日线数据重建，确保包含今日
    if daily:
        week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        this_week  = [d for d in daily if d["date"] >= week_start]
        if this_week:
            new_bar = {
                "date":     this_week[-1]["date"],
                "open":     this_week[0]["open"],
                "high":     max(d["high"]   for d in this_week),
                "low":      min(d["low"]    for d in this_week),
                "close":    this_week[-1]["close"],
                "volume":   sum(d["volume"] for d in this_week),
                "intraday": True,
            }
            if weekly and weekly[-1]["date"] >= week_start:
                weekly[-1] = new_bar   # 本周已有 → 替换为含今日数据的版本
            else:
                weekly.append(new_bar) # 本周缺失 → 追加

    return jsonify({"success": True, "daily": daily, "weekly": weekly})


@app.route("/api/stock/intraday/<code>")
def get_stock_intraday(code: str):
    """Return today's minute price series."""
    tc = _resolve_tc(code)

    # ── US stocks: use yfinance 1-minute data ─────────────────────────────────
    if tc.startswith("us"):
        try:
            import yfinance as yf
            t    = yf.Ticker(code)
            hist = t.history(period="1d", interval="1m", auto_adjust=True)
            if hist.empty:
                return jsonify({"success": False, "error": "No intraday data"})
            # Use yesterday's adjusted daily close as prev_close so it matches
            # the daily K-line chart (both use auto_adjust=True).
            prev_close = 0.0
            try:
                daily2 = t.history(period="2d", interval="1d", auto_adjust=True)
                if len(daily2) >= 2:
                    prev_close = round(float(daily2["Close"].iloc[-2]), 4)
                elif len(daily2) == 1:
                    prev_close = round(float(daily2["Close"].iloc[0]), 4)
            except Exception:
                pass
            if not prev_close:
                prev_close = float(t.fast_info.previous_close or 0)
            times, prices, volumes = [], [], []
            for ts, row in hist.iterrows():
                # ts is timezone-aware; format as HH:MM in local exchange time
                times.append(ts.strftime("%H:%M"))
                prices.append(round(float(row["Close"]), 4))
                volumes.append(int(row["Volume"]))
            return jsonify({"success": True, "times": times, "prices": prices,
                            "volumes": volumes, "prev_close": prev_close})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

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
        times, prices, cum_vols = [], [], []
        for item in raw:
            if isinstance(item, str):
                parts = item.split()
                if len(parts) >= 2:
                    t = parts[0].zfill(4)
                    times.append(f"{t[:2]}:{t[2:]}")
                    prices.append(float(parts[1]))
                    cum_vols.append(int(parts[2]) if len(parts) >= 3 else 0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                t = str(item[0]).zfill(4)
                times.append(f"{t[:2]}:{t[2:]}")
                prices.append(float(item[1]))
                cum_vols.append(int(item[2]) if len(item) >= 3 else 0)
        # diff cumulative volumes → per-minute volumes
        volumes = []
        for i, cv in enumerate(cum_vols):
            volumes.append(max(0, cv - (cum_vols[i - 1] if i > 0 else 0)))
        return jsonify({"success": True, "times": times, "prices": prices,
                        "volumes": volumes, "prev_close": prev_close})
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


def _build_trend_stock_list(market: str = "a") -> dict:
    """Build (or return cached) the trend-scan universe for the given market.

    market='a'  → A-shares ≥100亿 + curated ETFs  (cached 24 h)
    market='hk' → HK main-board stocks only        (cached 24 h)
    market='us' → S&P500 + supplements ≥300亿USD   (cached 1 h)
    """
    if market == "us":
        cache_key = "trend_stock_list_us_v1"
        cached = _get(cache_key, ttl=3600)
        if cached:
            return cached
        us = _fetch_us_stocks()
        for s in us:
            s["is_us"]  = True
            s["is_hk"]  = False
            s["is_etf"] = False
            s.setdefault("pe", 0.0)
        payload = {"success": True, "data": us, "total": len(us)}
        if us:
            _set(cache_key, payload)
        return payload

    if market == "hk":
        cache_key = "trend_stock_list_hk_v1"
        cached = _get(cache_key, ttl=86400)
        if cached:
            return cached
        hk = _fetch_hk_stocks()
        payload = {"success": True, "data": hk, "total": len(hk)}
        if hk:
            _set(cache_key, payload)
        return payload

    # A-share mode (default)
    cache_key = "trend_stock_list_a_v1"
    cached = _get(cache_key, ttl=86400)
    if cached:
        return cached
    a_stocks = [s for s in _fetch_a_stock_list() if s["市值亿"] >= 100.0]
    a_codes  = {s["代码"] for s in a_stocks}
    etfs     = [e for e in _MAJOR_ETFS if e["代码"] not in a_codes]
    payload = {
        "success": True,
        "data":    a_stocks + etfs,
        "total":   len(a_stocks) + len(etfs),
    }
    _set(cache_key, payload)
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
    body   = request.get_json(silent=True) or {}
    market = body.get("market", "a")
    try:
        cl = _build_trend_stock_list(market=market)
    except Exception as exc:
        return jsonify({"error": f"加载股票列表失败：{exc}"}), 400
    if not cl.get("data"):
        return jsonify({"error": "股票列表为空，请稍后重试"}), 400
    force = (request.args.get("force") == "1") or body.get("force", False)
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


@app.route("/api/trend/live_quotes", methods=["POST"])
def trend_live_quotes():
    """批量从腾讯实时行情拉取涨跌幅，用于盘中刷新趋势扫描结果。"""
    body     = request.get_json(silent=True) or {}
    tc_codes = body.get("tc_codes", [])
    if not tc_codes:
        return jsonify({})
    result = {}
    BATCH  = 200
    for i in range(0, len(tc_codes), BATCH):
        batch = tc_codes[i : i + BATCH]
        try:
            r = _sess.get(f"https://qt.gtimg.cn/q={','.join(batch)}", timeout=8)
            r.raise_for_status()
            for line in r.text.strip().split(";\n"):
                if '="' not in line:
                    continue
                var   = line.split("=")[0].strip()
                tc    = var[2:] if var.startswith("v_") else var
                inner = line.split('="', 1)[1].rstrip('";')
                flds  = inner.split("~")
                if len(flds) >= 5:
                    price = float(flds[3] or 0)
                    prev  = float(flds[4] or 0)
                    if price > 0 and prev > 0:
                        result[tc] = round((price - prev) / prev * 100, 2)
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/trend/screened")
def trend_screened():
    """返回与前端默认筛选条件完全一致的趋势选股结果（已过滤 + 评分排序）。

    接受的 query 参数（对应前端控件，均有默认值）：
      slope_min    斜率下限 %/周         默认 1.5
      slope_top    去掉斜率 top N%       默认 1
      sigma_max    σ_res 上限 %          默认 6
      slope_natr   斜率 ≤ N×ATR         默认 2
      mkt_min      市值下限 亿           默认 500
      mkt_max      市值上限 亿           默认不限
      excl_loss    排除亏损股 1/0        默认 1
      excl_reverse 排除反转股 1/0        默认 1
      penalty_20w  20周超涨惩罚 1/0      默认 1
    """
    with _trend_lock:
        results = list(_trend_scan.get("results", []))
        done    = _trend_scan.get("done", False)
    if not results:
        return jsonify({"ok": False, "error": "扫描尚未完成，请先触发 /api/trend/start"}), 400

    # ── 读取过滤参数 ────────────────────────────────────────────────────────────
    def _f(key, default):
        v = request.args.get(key, "")
        try: return float(v)
        except: return default

    slope_min    = _f("slope_min",    1.5)
    slope_top    = _f("slope_top",    1.0)
    sigma_max    = _f("sigma_max",    6.0)
    slope_natr   = _f("slope_natr",   2.0)
    mkt_min      = _f("mkt_min",      500.0)
    mkt_max      = _f("mkt_max",      float("inf"))
    excl_loss    = request.args.get("excl_loss",    "1") != "0"
    excl_reverse = request.args.get("excl_reverse", "1") != "0"
    penalty_20w  = request.args.get("penalty_20w",  "1") != "0"

    # 斜率 top N% 截止值
    slopes   = sorted(r.get("trend_slope", 0) for r in results)
    cutoff_i = int(len(slopes) * (1 - slope_top / 100))
    slope_max = slopes[max(0, cutoff_i)] if slopes else float("inf")

    # ── 过滤 ────────────────────────────────────────────────────────────────────
    def _pass(s):
        sl  = s.get("trend_slope", 0)
        sig = s.get("sigma_res", 999)
        mkt = s.get("市值亿", 0)
        pe  = s.get("pe", 0) or 0
        atr = s.get("atr_50d", 0)
        h50 = s.get("max_high_50d", 0)
        lc  = s.get("last_close", 0)
        is_etf = s.get("is_etf", False)

        if sl < slope_min:                                         return False
        if sl > slope_max:                                         return False
        if atr > 0 and sl > slope_natr * atr:                     return False
        if sig > sigma_max:                                        return False
        if not is_etf:
            if mkt < mkt_min:                                      return False
            if mkt_max < float("inf") and mkt > mkt_max:          return False
            if excl_loss and pe <= 0:                              return False
        if excl_reverse and h50 and lc and atr:
            if (1 - lc / h50) * 100 > 2 * atr:                    return False
        return True

    filtered = [s for s in results if _pass(s)]

    # ── 评分 ────────────────────────────────────────────────────────────────────
    def base_score(s):
        sl  = s.get("trend_slope", 0)
        sig = s.get("sigma_res", 0)
        if sl <= 0 or sig <= 0: return 0.0
        return (abs(sl) ** 1.1) * max(0.0, 8.0 - sig)

    def momentum_penalty(s):
        h50 = s.get("max_high_50d", 0)
        lc  = s.get("last_close", 0)
        atr = s.get("atr_50d", 0)
        if not h50 or not lc or not atr: return 1.0
        dd = (1 - lc / h50) * 100
        return 0.85 if dd > atr else 1.0

    def overext_penalty(s):
        if not penalty_20w: return 1.0
        r = (s.get("ret20w") or 0) / 100
        if r > 0.60: return 0.75
        if r > 0.40: return 0.85
        if r > 0.25: return 0.95
        return 1.0

    for s in filtered:
        s["_score"] = base_score(s) * momentum_penalty(s) * overext_penalty(s)

    filtered.sort(key=lambda s: -s["_score"])

    # ── 回撤标注 ────────────────────────────────────────────────────────────────
    def dd_warn(s):
        h50 = s.get("max_high_50d", 0)
        lc  = s.get("last_close", 0)
        atr = s.get("atr_50d", 0)
        if not h50 or not lc or not atr: return ""
        dd = (1 - lc / h50) * 100
        if dd > 3 * atr: return "⚠️"
        if dd > 2 * atr: return "⚡⚡"
        if dd > atr:     return "⚡"
        return ""

    out = []
    for i, s in enumerate(filtered, 1):
        h50 = s.get("max_high_50d", 0)
        lc  = s.get("last_close", 0)
        dd  = round((1 - lc / h50) * 100, 1) if h50 and lc else 0.0
        out.append({
            "rank":       i,
            "代码":       s.get("代码"),
            "名称":       s.get("名称"),
            "市值亿":     s.get("市值亿"),
            "斜率":       s.get("trend_slope"),
            "sigma_res":  s.get("sigma_res"),
            "得分":       round(s["_score"], 2),
            "PE":         s.get("pe"),
            "ATR%":       s.get("atr_50d"),
            "回撤%":      dd,
            "今日涨跌":   s.get("涨跌幅"),
            "ret20w":     s.get("ret20w"),
            "警示":       dd_warn(s),
            "行业":       s.get("industry_board"),
        })

    return jsonify({"ok": True, "done": done, "total": len(results),
                    "filtered": len(out), "results": out})


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
        cached = _get(f"hist_v4_{code}", ttl=86400)
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
        cached = _get(f"hist_v4_{code}", ttl=86400)
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

    hist_data = _get(f"hist_v4_{code}", ttl=86400)
    if not hist_data:
        return jsonify({"error": "请先在图表页加载该股票数据"}), 400

    pe_data = _get(f"pe_{code}", ttl=86400)

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
    cached    = _get(cache_key, ttl=43200)

    @stream_with_context
    def generate():
        if cached:
            yield f"data: {json.dumps({'text': cached, 'cached': True})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        buf: list[str] = []
        try:
            # ── ByteDance ModelHub（内网环境，支持 Google 搜索）─────────────
            if True:
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
                    model=model, stream=True,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "google_search"}],
                    tool_choice="auto", max_tokens=10000,
                    extra_headers={"X-TT-LOGID": f"dp-finder-{code}"},
                )
                in_search = False
                search_n  = 0
                tc_args: dict = {}
                fin_reason = None
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta  = chunk.choices[0].delta
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
                                    q = json.loads(tc_args[idx]).get("query", "")
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
    hist_data = _get(f"hist_v4_{code}", ttl=86400)
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
    cached = _get(cache_key, ttl=43200)

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
            # ── ByteDance ModelHub with vision（内网环境）────────────────────
            if True:
                endpoint = os.environ.get(
                    "MODELHUB_ENDPOINT",
                    "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl",
                )
                model = os.environ.get("MODELHUB_MODEL", "gemini-3-pro-preview-new")
                client = _AzureOpenAI(
                    azure_endpoint=endpoint, api_key=api_key,
                    api_version="2024-03-01-preview",
                    timeout=60,
                )
                messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": prompt},
                ]}]
                stream = client.chat.completions.create(
                    model=model, stream=True, messages=messages,
                    tools=[{"type": "google_search"}], tool_choice="auto",
                    max_tokens=10000,
                    extra_headers={"X-TT-LOGID": f"dp-finder-phase-{code}"},
                )
                in_search = False; search_n = 0; tc_args: dict = {}; fin_reason = None
                raw_chunks: list = []   # debug: keep last 5 chunks when buf is empty
                for chunk in stream:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    reason = chunk.choices[0].finish_reason
                    if not buf: raw_chunks.append(str(chunk)[:300])
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tc_args:
                                tc_args[idx] = ""
                                if not in_search:
                                    in_search = True; search_n += 1
                                    yield f"data: {json.dumps({'status': f'🔍 搜索中 ({search_n})…'})}\n\n"
                            if tc.function and tc.function.arguments:
                                tc_args[idx] += tc.function.arguments
                                try:
                                    q = json.loads(tc_args[idx]).get("query", "")
                                    if q:
                                        yield f"data: {json.dumps({'status': f'🔍 ({search_n}) {q}'})}\n\n"
                                except (json.JSONDecodeError, AttributeError):
                                    pass
                    # Some Gemini grounded responses put text in delta.content or message.content
                    content_text = delta.content
                    if not content_text and hasattr(chunk.choices[0], 'message'):
                        content_text = getattr(chunk.choices[0].message, 'content', None)
                    if content_text:
                        if in_search:
                            in_search = False
                            yield f"data: {json.dumps({'status': ''})}\n\n"
                        buf.append(content_text)
                        yield f"data: {json.dumps({'text': content_text})}\n\n"
                    if reason in ("stop", "length"):
                        fin_reason = reason; break
                # If Gemini returned nothing (e.g., search-only run with no text output),
                # retry once without google_search to force text generation.
                if not buf:
                    import sys
                    print(f"[phase] empty buf for {code}, retrying without google_search. chunks: {raw_chunks[-3:]}", file=sys.stderr)
                    yield f"data: {json.dumps({'status': '重试中（无搜索）…'})}\n\n"
                    stream2 = client.chat.completions.create(
                        model=model, stream=True, messages=messages,
                        max_tokens=10000,
                        extra_headers={"X-TT-LOGID": f"dp-finder-phase-{code}-retry"},
                    )
                    fin_reason = None
                    for chunk in stream2:
                        if not chunk.choices: continue
                        delta = chunk.choices[0].delta
                        reason = chunk.choices[0].finish_reason
                        content_text = delta.content
                        if not content_text and hasattr(chunk.choices[0], 'message'):
                            content_text = getattr(chunk.choices[0].message, 'content', None)
                        if content_text:
                            buf.append(content_text)
                            yield f"data: {json.dumps({'text': content_text})}\n\n"
                        if reason in ("stop", "length"):
                            fin_reason = reason; break
                if not buf:
                    yield f"data: {json.dumps({'error': 'AI 未返回分析内容，请重试'})}\n\n"
                    return
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
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
