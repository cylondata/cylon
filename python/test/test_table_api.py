from pytwisterx.table import PyTableAPI

table_api = PyTableAPI

print(table_api.row_count(b"id"), table_api.column_count(b"id"))