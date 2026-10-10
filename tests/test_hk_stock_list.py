import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from unittest.mock import Mock


class HKStockListTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text())
        names = {'_fetch_hk_stocks', '_fetch_hk_stocks_fallback', 'get_hk_stocks'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        for node in nodes:
            node.decorator_list = []
        self.env = dict(_sess=Mock(), _em_sess=Mock(), _get=Mock(return_value=None),
                        _set=Mock(), app=Mock(), time=Mock(),
                        ThreadPoolExecutor=ThreadPoolExecutor, jsonify=lambda v: v)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), self.env)

    def test_empty_primary_uses_fallback(self):
        self.env['_em_sess'].get.return_value.json.return_value = {'data': None}
        self.env['_fetch_hk_stocks_fallback'] = Mock(return_value=[{'代码': '00700'}])
        self.assertEqual(self.env['_fetch_hk_stocks'](), [{'代码': '00700'}])
        self.env['_set'].assert_called_once()

    def test_failure_is_not_cached_or_reported_as_success(self):
        self.env['_em_sess'].get.side_effect = TimeoutError()
        self.env['_fetch_hk_stocks_fallback'] = Mock(side_effect=TimeoutError())
        payload, status = self.env['get_hk_stocks']()
        self.assertEqual(status, 503)
        self.assertFalse(payload['success'])
        self.env['_set'].assert_not_called()

    def test_actual_page_size_and_dict_format(self):
        self.env['_em_sess'].get.return_value.json.side_effect = [
            {'data': {'total': 2, 'diff': {'0': {'f12': '00700', 'f20': 1e12, 'f5': 100000}}}},
            {'data': {'total': 2, 'diff': [{'f12': '00005', 'f20': 1e11, 'f5': 100000}]}}]
        self.assertEqual(len(self.env['_fetch_hk_stocks']()), 2)
        self.assertEqual(self.env['_em_sess'].get.call_count, 2)

    def test_tencent_units_and_fund_exclusion(self):
        def quote(code, kind):
            fields = ['0'] * 78
            for index, value in {1: 'Test', 2: code, 3: '424.8', 6: '200000',
                                 32: '3.26', 39: '15.52', 45: '38625.3886',
                                 63: kind, 75: 'HKD'}.items():
                fields[index] = value
            return f'v_hk{code}="' + '~'.join(fields) + '";'
        count = Mock()
        count.json.return_value = '2'
        page = Mock()
        page.json.return_value = [{'symbol': '00700', 'volume': '200000'},
                                  {'symbol': '02800', 'volume': '200000'}]
        quotes = Mock(text=quote('00700', 'GP') + '\n' + quote('02800', 'GP-FUND'))
        self.env['_sess'].get.side_effect = [count, page, quotes]
        data = self.env['_fetch_hk_stocks_fallback']()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['市值亿'], 38625.4)
        self.assertEqual(data[0]['最新价'], 424.8)


if __name__ == '__main__':
    unittest.main()
