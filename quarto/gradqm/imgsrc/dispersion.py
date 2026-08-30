import numpy as np
import matplotlib.pyplot as plt

pvec = np.arange(-20,20)
N = len(pvec)
k = 2*np.pi* pvec/N
E = -2 * np.cos(k)
v = 2*np.sin(k)

plt.subplot(1,2,1)
plt.plot(k, E, 'o')
plt.xlim(-np.pi, np.pi)

plt.subplot(1,2,2)
plt.plot(k, v, 'o')
plt.xlim(-np.pi, np.pi)

plt.show()
