from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import tmm
import sys

## Plot transmittance vs E
def resonance_demo_1(U, Vb, a, b, Emax):
    Evec = linspace(0.01, Emax, 5000)

    psim = empty(len(Evec), dtype=complex)
    psip = empty(len(Evec), dtype=complex)

    L, V = array([b-a, 2*a, b-a]), array([Vb, Vb-U, Vb])
    for jj in range(len(Evec)):
        k = sqrt(2*Evec[jj])
        M = tmm.transfer_matrix(L, V, 0.0, Evec[jj])
        psim[jj] = -M[1,0]/M[1,1]*exp(-2j*k*b)
        psip[jj] = M[0,0] * exp(-2j*k*b) + M[0,1]*psim[jj]

    plt.plot(Evec, abs(psip)**2)
    plt.xlim(0, Emax)
    plt.ylim(0,1.05)
    plt.show()

resonance_demo_1(20.0, 30.0, 1.0, 1.4, 70.0)


## Plot resonance wavefunctions
def resonance_demo_2():
    from scipy.integrate import trapz
    U, Vb = 20.0, 30.0
    a, b = 1.0, 1.2
    nstates = 3
    ## From call to Ebound
    Eres = [10.918, 13.646, 18.092, 24.002, 29.973];

    ## Plot resonance wavefunctions (left input = 1)
    L, V = array([b-a, 2*a, b-a]), array([Vb, Vb-U, Vb])
    for n in range(nstates):
        S = tmm.scattering_matrix(L, V, 0.0, Eres[n])
        components = array([1.0, S[0,0]])
        x, psi = tmm.wavefunction(L, V, Eres[n], components)
        plt.subplot(2, nstates, n+1)
        plt.plot(x-b, abs(psi)**2)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0)

    ## Plot the corresponding square well bound states
    L, V = array([2*a]), array([-U])
    Ebound = tmm.bound_state_energies(L, V)
    inputwave = array([0, 1.0], dtype=complex)
    for n in range(nstates):
        x, psi = tmm.wavefunction(L, V, Ebound[n], inputwave)
        ## Normalize
        normalization = trapz(abs(psi)**2, x)
        psi /= sqrt(normalization)
        plt.subplot(2, nstates, n+1+nstates)
        plt.plot(x-a, abs(psi)**2)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0,1)
    plt.show()

## Plot resonance width vs b, and compare to Fermi's Golden Rule.
def resonance_demo_3():
    U, Vb = 20.0, 30.0
    a     = 1.0
    V = array([Vb, Vb-U, Vb])

    ## For this E resolution, we can let b range over ~ [1.1, 1.8]
    Evec = linspace(10.6, 11.2, 5000)
    bvec = linspace(1.1, 1.6, 20)
    T    = empty(len(Evec))
    fwhm = empty(len(bvec))

    def transmittance(E, dT=0.0):
        k  = sqrt(2*E)
        M  = tmm.transfer_matrix(L, V, 0.0, E)
        fm = -M[1,0]/M[1,1]*exp(-2j*k*b)
        return abs(M[0,0] * exp(-2j*k*b) + M[0,1]*fm)**2 - dT

    for ii in range(len(bvec)):
        b = bvec[ii]
        L = array([b-a, 2*a, b-a])
        for jj in range(len(Evec)):
            T[jj] = transmittance(Evec[jj])
        ## Locate energies where T = 0.5
        dT  = T - 0.5
        idx = nonzero(dT[:-1] * dT[1:] < 0)[0]
        assert len(idx) == 2
        Eres = empty(2)
        for n in range(2):
            jj   = idx[n]
            Eres[n] = brentq(transmittance, Evec[jj-1], Evec[jj+2], (0.5,))
        fwhm[ii] = Eres[1] - Eres[0]
        sys.stdout.write('\r' + str(ii+1) + '/' + str(len(bvec)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    plt.plot(bvec-a, fwhm, 'o')

    ## Fermi's Golden Rule prediction
    L, V = array([2*a]), array([-U])
    E0  = tmm.bound_state_energies(L, V)[0]
    q   = sqrt(2*(E0+U))
    et  = sqrt(-2*E0)
    k   = sqrt(2*(E0+Vb))
    DOS = sqrt(2/(E0+Vb))
    Bsq = exp(2*et*a)/a/((1+sin(2*q*a)/2/q/a)/cos(q*a)**2 + 1/et/a)

    bb = linspace(1.05, 1.8, 200)
    overlap = (et*cos(k*bb) - k*sin(k*bb))**2 * exp(-2*et*bb) / (2*pi)
    K = 2*pi * Bsq * overlap * DOS

    plt.plot(bb-a, K)
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.35)
    plt.show()

# resonance_demo_1()
