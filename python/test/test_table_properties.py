# import pandas as pd
#
# customer = pd.DataFrame({
#     'id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     'name': ['Olivia', 'Aditya', 'Cory', 'Isabell', 'Dominic', 'Tyler', 'Samuel', 'Daniel', 'Jeremy'],
#     'age': [20, 25, 15, 10, 30, 65, 35, 18, 23],
#     'Product_ID': [101, 0, 106, 0, 103, 104, 0, 0, 107],
#     'Purchased_Product': ['Watch', 'NA', 'Oil', 'NA', 'Shoes', 'Smartphone', 'NA', 'NA', 'Laptop'],
#     'City': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Chennai', 'Delhi', 'Kolkata', 'Delhi', 'Mumbai']
# })
# product = pd.DataFrame({
#     'Product_ID': [1011, 1021, 1031, 1041, 1051, 1061, 1071],
#     'Product_ID': [101, 102, 103, 104, 105, 106, 107],
#     'Product_name': ['Watch', 'Bag', 'Shoes', 'Smartphone', 'Books', 'Oil', 'Laptop'],
#     'Category': ['Fashion', 'Fashion', 'Fashion', 'Electronics', 'Study', 'Grocery', 'Electronics'],
#     'Price': [299.0, 1350.50, 2999.0, 14999.0, 145.0, 110.0, 79999.0],
#     'Seller_City': ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Delhi', 'Chennai', 'Bengalore'],
# })
#
# newdf = pd.merge(product, customer, on='Product_ID')
#
# print(newdf)

from pycylon import Table
from pycylon.csv import csv_reader
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None)

tb: Table = csv_reader.read(ctx, '/tmp/user_usage_tm_1.csv', ',')

print("Table Column Names")
print(tb.column_names)


print("Table Schema")
print(tb.schema)

ctx.finalize()
