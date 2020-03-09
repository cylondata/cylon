import pytwisterx as tx

r = tx.Request('m1', 4, 'Hello')

assert r.getName() == 'm1'
assert r.getBufSize() == 4
assert r.getMessage() == 'Hello'




