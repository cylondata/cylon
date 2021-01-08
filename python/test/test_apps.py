from pycylon import CylonContext
from pycylon import Table
import numpy as np
import pandas as pd

ctx = CylonContext(config=None, distributed=False)

pdf = pd.DataFrame([np.nan] * 5, dtype='object')
target = 'AUC'
records = 10

a = np.random.randint(5, records)

tb = Table.from_pydict(ctx, {'Source': np.random.randint(5, size=records).tolist(),
                             'Drug1': np.random.randint(5, size=records).tolist(),
                             'Drug2': np.random.randint(5, size=records).tolist(),
                             'Sample': np.random.randint(5, size=records).tolist(),
                             'AUC': np.random.randint(5, size=records).tolist()})

print(tb)

#tb['c'] = Table.from_pandas(ctx, pdf)

print(tb)


df = tb.to_pandas()

df_sum = df.groupby('Source').agg({target: 'count', 'Sample': 'nunique',
                                       'Drug1': 'nunique', 'Drug2': 'nunique'})

print(df_sum)