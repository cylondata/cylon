

def resolve_column_index_from_column_name(column_name, table) -> int:
    index = None
    for idx, col_name in enumerate(table.column_names):
        if column_name == col_name:
            return idx
    if index is None:
        raise ValueError(f"Column {column_name} does not exist in the table")