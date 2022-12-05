#%%
from functools import cache
from scipy.sparse import random,csr_matrix
from scipy.sparse.linalg import expm_multiply
from time import time
import matplotlib.pyplot as plt
import numpy as np

#%%
ns= list(range(15, 30))
ms = [m*50 for m in range(1, 10)]

number_terms = 15

row_ind = np.arange(2**15)
col_ind = np.random.randint(0, 2**15, 2**15)
data = np.random.rand(2**15)
#%%


times = []
for n in ns:
    print(n)
    rand = csr_matrix((data, (2**(n-number_terms)*row_ind, 2**(n-number_terms)*col_ind)), shape=(2**n, 2**n))
    vector = np.random.rand(2**n)
    start = time()
    _ = expm_multiply(rand,vector)
    end = time()
    print(end-start)
    times.append(end-start)

#%%
print(ns,times)
plt.plot(ns, np.log(times))


#%%