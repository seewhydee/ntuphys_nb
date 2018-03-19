from scipy import *
import matplotlib.pyplot as plt

## Lithium:
## Phys. Rev. A 26, 3351 (1982)

li_lambda = 670.8e-9
li_tau = 27.29e-9
li_dtau = 0.04e-9

na_lambda = 589.6e-9
na_tau = 16.40e-9
na_dtau = 0.03e-9

## Hydrogen: Phys. Rev. 148, 1, 1966

## 1s-2p
h_2p_lambda = 1215.6e-10
h_2p_tau = 1.6e-9
h_2p_dtau = 0.004e-9

h_3p_lambda = 1025.6e-10
h_3p_tau = 5.58e-9
h_3p_dtau = 0.13e-9

plt.errorbar(array([li_lambda, na_lambda, h_2p_lambda, h_3p_lambda]) * 1e9,
             array([li_tau, na_tau, h_2p_tau, h_3p_tau]) * 1e9,
             yerr=[li_dtau, na_dtau, h_2p_dtau, h_3p_dtau],
             fmt='o')

c = 299792458
alpha = 0.0072973525664
d = 1e-10;
lvec = linspace(0.01e-6, 1e-6, 100)
k = 2*pi/lvec
omega = c * k
tau = 3*c*c/(4*alpha*omega**3 * d*d)

plt.plot(lvec*1e9, tau*1e9, '--')

plt.xlim(0, 760)
plt.ylim(0, 40)
plt.show()
