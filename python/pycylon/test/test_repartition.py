from pycylon import CylonEnv
from utils import create_df,assert_eq
from pycylon.net import MPIConfig
import random

"""
Run test:
>> pytest -q python/pycylon/test/test_repartition.py
"""

def test_repartition():
    env=CylonEnv(config=MPIConfig())
    df1, _ = create_df([random.sample(range(10, 300), 50),
                            random.sample(range(10, 300), 50),
                            random.sample(range(10, 300), 50)])
    df2 = df1.repartition([50], None, env=env)
    assert_eq(df1, df2)
