from scipy import *
from scipy.special import spherical_jn, spherical_yn, lpmv
from scipy.stats import uniform
import matplotlib.pyplot as plt
import sys

## Scattering amplitude versus theta
def spherical_well_f_theta(E, U, R, thetavec, lmax=30):
    lvec = arange(lmax)

    P = zeros((len(lvec), len(thetavec)))
    for jj in range(len(lvec)):
        P[jj,:] = lpmv(0, lvec[jj], cos(thetavec))

    kR = sqrt(2*E)*R
    qR = sqrt(2*(E+U))*R

    jl  = spherical_jn(lvec, qR)
    jlp = spherical_jn(lvec, qR, derivative=True)
    hl  = spherical_jn(lvec, kR) + 1j*spherical_yn(lvec, kR)
    hlp = spherical_jn(lvec, kR, True) + 1j*spherical_yn(lvec, kR, True)

    delt = 0.5*pi + angle((kR*hlp*jl - qR*hl*jlp)/R)
    coef = (exp(2j*delt)-1) * (2*lvec+1)

    return (-0.5j*R/kR) * dot(coef, P)

## Scattering amplitude versus E
def spherical_well_f_E(Evec, U, R, theta, lmax=30):
    costheta = cos(theta)

    P = lpmv(0, arange(lmax), costheta)
    f = zeros(len(Evec), dtype=complex)
    for ll in range(lmax):
        kR = sqrt(2*Evec)*R
        qR = sqrt(2*(Evec+U))*R

        jl  = spherical_jn(ll, qR)
        jlp = spherical_jn(ll, qR, derivative=True)
        hl  = spherical_jn(ll, kR) + 1j*spherical_yn(ll, kR)
        hlp = spherical_jn(ll, kR, True) + 1j*spherical_yn(ll, kR, True)

        delt = 0.5*pi + angle((kR*hlp*jl - qR*hl*jlp)/R)
        f += (-0.5j*R/kR) * (exp(2j*delt)-1) * (2*ll+1) * P[ll]
    return f

## Monte Carlo calculation of scattering amplitude versus E
def spherical_well_f_E_mc(Evec, U, R, theta, nmc=10000):
    rdist = uniform(loc=-R, scale=(2*R))
    Rsq   = R*R

    f1    = zeros(len(Evec), dtype=complex)
    f2    = zeros(len(Evec), dtype=complex)
    coef1 = -U*(2*R)**3/(2*pi*nmc)
    coef2 = -U*U*(2*R)**6/(2*pi*nmc)
    for n in range(len(Evec)):
        k   = sqrt(2*Evec[n])
        kfx = k*sin(theta)
        kfz = k*cos(theta)
        ## First Born approximation
        for jj in range(nmc):
            r = rdist.rvs(3)
            if dot(r,r) < Rsq:
                f1[n] += exp(1j*(k*r[2] - kfx*r[0] - kfz*r[2]))
        f1[n] *= coef1
        ## Second Born approximation
        for jj in range(nmc):
            r1, r2 = rdist.rvs(3), rdist.rvs(3)
            if dot(r1,r1) < Rsq and dot(r2,r2) < Rsq:
                dr = r1 - r2
                dr = sqrt(dot(dr,dr))
                G = -exp(1j*k*dr)/(2*pi*dr)  ## SIGN??
                f2[n] += exp(1j*(k*r2[2] - kfx*r1[0] - kfz*r1[2])) * G
        f2[n] *= coef2
        sys.stdout.write('\r' + str(n+1) + '/' + str(len(Evec)))
    sys.stdout.flush()

    return f1, f1+f2

## Plot angular dependence of scatterer
def spherical_well_angle_demo(E=0.10, U=1.0, R=1.0):
    thetavec = linspace(0,pi, 700)
    f = spherical_well_f_theta(E, U, R, thetavec)
    plt.plot(thetavec, abs(f)**2)
    plt.xlim(0,pi)
    plt.ylim(0)
    plt.show()

## Plot E dependence of scatterer
def spherical_well_E_demo():
    theta = pi/2
    U, R  = 0.1, 1.0
    Evec  = linspace(1e-6, 10, 1000)
    Evec2 = linspace(1e-6, 10, 10)

    f = spherical_well_f_E(Evec, U, R, theta)
    f_B1, f_B2 = spherical_well_f_E_mc(Evec2, U, R, theta)

    plt.plot(Evec, abs(f)**2)
    plt.plot(Evec2, abs(f_B1)**2, 'o')
    plt.plot(Evec2, abs(f_B2)**2, 'x')
    plt.xlim(0, Evec[-1])
    plt.ylim(0)
    plt.show()

spherical_well_E_demo()
