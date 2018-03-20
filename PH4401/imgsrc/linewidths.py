from scipy import *
import matplotlib.pyplot as plt

## https://physics.nist.gov/PhysRefData/ASD/lines_form.html

## 2p-1s, 3p-1s
H_lambda = array([121.567]) * 1e-9
H_A = array([6.2648e8])

## 2p-2s
Li_lambda = array([670.791]) * 1e-9
Li_A = array([3.689e7])

## 3p-3s
Na_lambda = array([588.995]) * 1e-9
Na_A = array([6.16e7])

plt.plot(H_lambda * 1e9, H_A, 'ro')
plt.plot(Li_lambda * 1e9, Li_A, 'go')
plt.plot(Na_lambda * 1e9, Na_A, 'bo')

c = 299792458
alpha = 0.0072973525664
d = 1e-10;
lvec = linspace(5e-8, 1e-6, 100)
k = 2*pi/lvec
omega = c * k
tau = 3*c*c/(4*alpha*omega**3 * d*d)

plt.plot(lvec * 1e9, 1/tau, '--')

plt.xlim(0, 800)
plt.ylim(0, 1e9)
plt.show()
