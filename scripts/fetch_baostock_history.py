"""Isolate BaoStock's process-global socket and bound it from the parent process."""
import json
import sys

import baostock as bs


def main():
    code, start, end = sys.argv[1:]
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(login.error_msg)
    try:
        result = bs.query_history_k_data_plus(
            code, 'date,close', start_date=start, end_date=end,
            frequency='d', adjustflag='1')
        rows = []
        while result.error_code == '0' and result.next():
            rows.append(result.get_row_data())
        if result.error_code != '0' or not rows:
            raise RuntimeError(result.error_msg or 'empty BaoStock history')
        print('HISTORY_JSON:' + json.dumps(rows))
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
