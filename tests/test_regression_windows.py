import ast
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np


class RegressionWindowTests(unittest.TestCase):
    def payload(self, end):
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'_ols', '_compute_regression_payload'}]
        dates = [(end - timedelta(days=500 - i)).strftime('%Y-%m-%d') for i in range(501)]
        prices = np.exp(np.arange(501) * .001 + 2).tolist()
        env = dict(np=np, datetime=datetime, timedelta=timedelta,
                   _get=Mock(return_value={'success': True}), _set=Mock(),
                   _MAJOR_ETFS=[], _fetch_kline_hfq=Mock(return_value=(dates, prices)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), env)
        return env['_compute_regression_payload']('hk00700', '00700')

    def test_six_month_stats_and_aligned_bands(self):
        d = self.payload(datetime(2026, 10, 9))
        self.assertEqual(d['split_6m_date'], '2026-04-09')
        start = d['split_6m_idx']
        self.assertEqual(d['stats_6m']['total_days'], len(d['dates']) - start)
        self.assertAlmostEqual(d['stats_6m']['r_squared'], 1)
        self.assertAlmostEqual(d['stats_6m']['annual_growth'], np.expm1(.252) * 100, places=2)
        for key in ('reg_6m', 'lower_band_6m', 'upper_band_6m'):
            self.assertEqual(len(d[key]), len(d['dates']))
            self.assertTrue(all(v is None for v in d[key][:start]))
            self.assertTrue(all(v > 0 for v in d[key][start:]))

    def test_calendar_month_end_clamping(self):
        self.assertEqual(self.payload(datetime(2024, 8, 31))['split_6m_date'], '2024-02-29')
        self.assertEqual(self.payload(datetime(2025, 8, 31))['split_6m_date'], '2025-02-28')


if __name__ == '__main__':
    unittest.main()
