import ast
import logging
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock


class StockListTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text())
        names = {'_fetch_a_list_page', '_fetch_a_stock_list',
                 '_save_a_list_snapshot', '_load_a_list_snapshot'}
        module = ast.Module(body=[n for n in tree.body
                                 if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[])
        self.env = dict(time=time, threading=threading, _cache={},
                        _cache_lock=threading.Lock(), _redis=None,
                        _em_sess=Mock(), _get=Mock(return_value=None), _set=Mock(),
                        app=Mock(logger=logging.getLogger('test')))
        exec(compile(module, 'app.py', 'exec'), self.env)

    def test_host_retry_and_dict_response(self):
        response = Mock()
        response.json.return_value = {'data': {'total': 1, 'diff': {'0': {'f12': '600000'}}}}
        self.env['_em_sess'].get.side_effect = [TimeoutError(), response]
        data = self.env['_fetch_a_list_page']({})
        self.assertEqual(data['diff'], [{'f12': '600000'}])
        self.assertEqual(self.env['_em_sess'].get.call_count, 2)

    def test_complete_list_saved(self):
        self.env['_fetch_a_list_page'] = Mock(return_value={
            'total': 1, 'diff': [{'f12': '600000', 'f14': 'Test', 'f20': 1e11}]})
        result = self.env['_fetch_a_stock_list']()
        self.assertEqual(result[0]['市值亿'], 1000)
        self.env['_set'].assert_called_once()
        self.assertIn('a_stock_list_last_good', self.env['_cache'])

    def test_failed_pagination_uses_snapshot_not_partial_results(self):
        self.env['_save_a_list_snapshot']([{'代码': '600001'}])
        self.env['_fetch_a_list_page'] = Mock(side_effect=[
            {'total': 2, 'diff': [{'f12': '600000', 'f20': 1e11}]}, TimeoutError()])
        result = self.env['_fetch_a_stock_list']()
        self.assertEqual(result[0]['代码'], '600001')
        self.assertIn('_stale_as_of', result[0])
        self.env['_set'].assert_not_called()

    def test_expired_snapshot_not_used(self):
        self.env['_cache']['a_stock_list_last_good'] = {
            'ts': time.time() - 86400 * 8, 'data': [{'代码': '600000'}]}
        self.assertEqual(self.env['_load_a_list_snapshot'](), [])

    def test_empty_response_retried_bounded(self):
        self.env['_em_sess'].get.return_value.json.return_value = {'data': None}
        with self.assertRaises(RuntimeError):
            self.env['_fetch_a_list_page']({})
        self.assertEqual(self.env['_em_sess'].get.call_count, 3)


if __name__ == '__main__':
    unittest.main()
