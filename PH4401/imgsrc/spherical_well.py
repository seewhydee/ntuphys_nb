from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import spherical_jn, spherical_kn, spherical_yn, lpmv
from scipy.stats import uniform
import sys

## Find and report the energies of a spherical well
def bound_state_energies(a, V0, l):
    E = linspace(-0.9999*V0, -0.00001*V0, 300)
    f = empty(len(E))

    def fun(Ecomplex):
        E = Ecomplex[0] + 1j*Ecomplex[1]
        q = sqrt(2*(E+V0))
        g = sqrt(-2*E)
        y = q*spherical_jn(l, q*a, True)*spherical_kn(l,g*a) \
          - g*spherical_kn(l, g*a, True)*spherical_jn(l,q*a)
        return array([real(y), imag(y)])

    for n in range(len(E)):
        q = sqrt(2*(E[n]+V0))
        g = sqrt(-2*E[n])
        y = fun([E[n],0.0])
        f[n] = log(y[0]**2+y[1]**2)
    ## Look for minima of log|f|
    idx = nonzero((f[:-2]>f[1:-1]) * (f[1:-1] < f[2:]))[0]
    if len(idx) == 0:
        return []
    Ebound = E[idx + 1]

    ## Refine the guesses
    for n in range(len(Ebound)):
        res = root(fun, [Ebound[n], 1e-4])
        assert res.success
        Ebound[n] = (res.x)[0]
    return Ebound

# print(bound_state_energies(1.0, 20.0, 1))

## Plot the bound state energies of a spherical well.
def spherical_well_demo():
    a  = 1.0
    nb = 10

    lvec = [0, 1, 2, 3, 4, 5]
    cols = ['r', 'y', 'g', 'c', 'b', 'm']

    for ll in range(len(lvec)):
        l = lvec[ll]

        V0vec = linspace(0.05, 20.0, 200)
        E = zeros((nb, len(V0vec)))
        for jj in range(len(V0vec)):
            Eb = sort(bound_state_energies(a, V0vec[jj], l))
            if len(Eb) > 0:
                E[:len(Eb),jj] = Eb
            sys.stdout.write('\rl = ' + str(ll) + ', ' + str(jj+1) + '/' + str(len(V0vec)))
        sys.stdout.flush()
        sys.stdout.write('\n')

        for n in range(nb):
            idx = nonzero(E[n])[0]
            if len(idx) > 0:
                plt.plot(-V0vec[idx], E[n,idx], cols[ll])
    plt.xlim(-V0vec[-1], 0)
    plt.ylim(-20, 0)
    plt.show()

## Compute scattering amplitude versus theta
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

## Compute scattering amplitude versus E
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

    ## Pre-factors for the scattering amplitude calculation, including
    ## (i) the 1/(2*pi) in the definition of f, (ii) appropriate
    ## multiples of the V factor (where V = -U), (iii) the volume of
    ## the Monte Carlo integration domain, (iv) the inverse of the
    ## number of Monte Carlo samples (so as to get the mean), and (v)
    ## the constant prefactor in the propagator (for the second Born
    ## approximation integral).
    coef1 = -(-U) * (2*R)**3 / (2*pi) / nmc
    coef2 = -(-U)**2 * (2*R)**6 / (2*pi) / nmc * (-1/(2*pi))
    for n in range(len(Evec)):
        k   = sqrt(2*Evec[n])
        kfx = k*sin(theta)
        kfz = k*cos(theta)
        ## First Born approximation
        for jj in range(nmc):
            r = rdist.rvs(3)
            if dot(r,r) < Rsq:
                f1[n] += exp(1j*(k*r[2] - kfx*r[0] - kfz*r[2]))
        ## Second Born approximation
        for jj in range(nmc):
            r1, r2 = rdist.rvs(3), rdist.rvs(3)
            if dot(r1,r1) < Rsq and dot(r2,r2) < Rsq:
                dr = r1 - r2
                dr = sqrt(dot(dr,dr))
                ## Constant factors in G are absorbed into coef2
                G = exp(1j*k*dr)/dr
                f2[n] += exp(1j*(k*r2[2] - kfx*r1[0] - kfz*r1[2])) * G
        sys.stdout.write('\r' + str(n+1) + '/' + str(len(Evec)))
    sys.stdout.flush()
    f1 *= coef1
    f2 *= coef2
    return f1, f2

## Plot angular dependence of scattering amplitudes
def spherical_well_angle_demo(E=0.10, U=1.0, R=1.0):
    thetavec = linspace(0,pi, 700)
    f = spherical_well_f_theta(E, U, R, thetavec)
    plt.plot(thetavec, abs(f)**2)
    plt.xlim(0,pi)
    plt.ylim(0)
    plt.show()

## Plot E dependence of scattering amplitudes, and compare to Born
## series (from Monte Carlo integration).
def spherical_well_E_demo():
    theta = pi/2
    U, R  = 0.1, 1.0
    Evec  = linspace(1e-6, 10, 1000)
    Evec2 = linspace(1e-6, 10, 25)

    f = spherical_well_f_E(Evec, U, R, theta)
    f1, f2 = spherical_well_f_E_mc(Evec2, U, R, theta)

    plt.plot(Evec, abs(f)**2)
    plt.plot(Evec2, abs(f1)**2, 'o-')
    plt.plot(Evec2, abs(f1 + f2)**2, 'x-')
    plt.xlim(0, Evec[-1])
    plt.ylim(0)
    plt.show()

spherical_well_E_demo()
