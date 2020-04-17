from pytwisterx.api.comms.channel import TxRequest
import numpy as np
buf = np.array([1,2,3,4,5,6,7,8], dtype=np.int32)
tx = TxRequest(10, buf, 8, np.array([1,2,3,4], dtype=np.int32), 4)

print(tx.target, tx.buf, tx.header, tx.headerLength, tx.length)

print(type(tx.buf), type(tx.header))

print(tx.to_string(b'int', 32))