from scipy import *

random.seed(1)

L = arange(0.1,0.7,0.06)
N = len(L)
g = 9.98
T = 2*pi*sqrt(L/g) + 0.05*randn(N)
dT = 0.05

Tsq = T*T
Tsq_error = 2 * T * dT

import matplotlib.pyplot as plt
plt.errorbar(L, Tsq, yerr=Tsq_error, fmt='o')

L2 = linspace(0, 0.7, 100)
T2 = 2*pi*sqrt(L2/g)
plt.plot(L2, T2*T2)
plt.show()


print(L)
print(T)
