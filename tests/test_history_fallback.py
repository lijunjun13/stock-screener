import ast
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np


class HistoryFallbackTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_fetch_kline_hfq')
        self.env = dict(datetime=datetime, timedelta=timedelta, np=np, time=Mock(),
                        app=Mock(), _get=Mock(return_value=None), _set=Mock(),
                        _fetch_kline_baostock=Mock(side_effect=RuntimeError('unavailable')),
                        _em_sess=Mock(), _fetch_kline_hfq_tencent=Mock(side_effect=RuntimeError('501')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), self.env)

    def test_full_interval_hfq_and_cached_result(self):
        date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        self.env['_em_sess'].get.return_value.json.return_value = {
            'data': {'klines': [f'{date},10,11,12,9,100']}}
        self.assertEqual(self.env['_fetch_kline_hfq']('sh600030'), ([date], [11]))
        params = self.env['_em_sess'].get.call_args.kwargs['params']
        self.assertEqual(params['fqt'], '2')
        self.assertEqual(params['secid'], '1.600030')
        self.env['_set'].assert_called_once()

    def test_primary_success_does_not_use_fallback(self):
        self.env['_fetch_kline_hfq_tencent'] = Mock(return_value=(['2026-10-09'], [100]))
        self.assertEqual(self.env['_fetch_kline_hfq']('sh600030')[1], [100])
        self.env['_em_sess'].get.assert_not_called()

    def test_empty_fallback_retries_without_caching(self):
        self.env['_em_sess'].get.return_value.json.return_value = {'data': None}
        with self.assertRaises(RuntimeError):
            self.env['_fetch_kline_hfq']('sh600030')
        self.assertEqual(self.env['_em_sess'].get.call_count, 3)
        self.env['_set'].assert_not_called()

    def test_baostock_used_when_both_http_sources_fail(self):
        self.env['_em_sess'].get.side_effect = TimeoutError()
        self.env['_fetch_kline_baostock'] = Mock(return_value=(['2026-10-09'], [129.38]))
        self.assertEqual(self.env['_fetch_kline_hfq']('sh600030')[1], [129.38])
        self.env['_set'].assert_called_once()


if __name__ == '__main__':
    unittest.main()
