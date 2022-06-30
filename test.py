import numpy as np
import time

query_3d = np.random.randn(1, 512)
feats_2d = np.random.randn(150000, 512)

t0 = time.time()

res = np.dot(feats_2d, query_3d.T)
print(res.shape)
print(f"cost time: {time.time() - t0}")
