from scipy import *
from scipy.special import spherical_jn, spherical_yn, lpmv
import matplotlib.pyplot as plt

## Plot angular dependence of scatterer
def spherical_scattering_angle_demo(E=0.10, U=1.0, R=1.0):
    lmax = 30
    theta = linspace(0,pi, 700)
    lvec = arange(lmax)

    P = zeros((len(lvec), len(theta)))
    for jj in range(len(lvec)):
        P[jj,:] = lpmv(0, lvec[jj], cos(theta))

    kR = sqrt(2*E)*R
    qR = sqrt(2*(E+U))*R

    jl  = spherical_jn(lvec, qR)
    jlp = spherical_jn(lvec, qR, derivative=True)
    hl  = spherical_jn(lvec, kR) + 1j*spherical_yn(lvec, kR)
    hlp = spherical_jn(lvec, kR, True) + 1j*spherical_yn(lvec, kR, True)

    delt = 2*angle((kR*hlp*jl - qR*hl*jlp)/R)
    coef = (exp(1j*delt)+1) * (2*lvec+1)

    f = (0.5j*R/kR) * dot(coef, P)
    plt.plot(theta, abs(f)**2)
    plt.xlim(0,pi)
    plt.ylim(0)
    plt.show()

spherical_scattering_angle_demo()
