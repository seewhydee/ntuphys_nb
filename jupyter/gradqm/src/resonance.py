import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import tmm
import sys

## Plot transmittance vs E
def resonance_demo_1(U, Vb, a, b, Emax):
    Evec = np.linspace(0.01*Emax, Emax, 5000)

    psim = np.empty(len(Evec), dtype=complex)
    psip = np.empty(len(Evec), dtype=complex)

    L, V = np.array([b-a, 2*a, b-a]), np.array([Vb, Vb-U, Vb])
    for jj in range(len(Evec)):
        k = np.sqrt(2*Evec[jj])
        M = tmm.transfer_matrix(L, V, 0.0, Evec[jj])
        psim[jj] = -M[1,0]/M[1,1]*np.exp(-2j*k*b)
        psip[jj] = M[0,0] * np.exp(-2j*k*b) + M[0,1]*psim[jj]

    plt.plot(Evec, abs(psip)**2)
    plt.xlim(0, Emax)
    plt.ylim(0,1.05)
    plt.show()

## Plot resonance wavefunctions
def resonance_demo_2():
    from scipy.integrate import trapz
    U, Vb = 20.0, 30.0
    a, b = 1.0, 1.2
    nstates = 3
    ## From call to Ebound
    Eres = [10.918, 13.646, 18.092, 24.002, 29.973];

    ## Plot resonance wavefunctions (left input = 1)
    L, V = np.array([b-a, 2*a, b-a]), np.array([Vb, Vb-U, Vb])
    for n in range(nstates):
        S = tmm.scattering_matrix(L, V, 0.0, Eres[n])
        components = np.array([1.0, S[0,0]])
        x, psi = tmm.wavefunction(L, V, Eres[n], components)
        plt.subplot(2, nstates, n+1)
        plt.plot(x-b, abs(psi)**2)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0)

    ## Plot the corresponding square well bound states
    L, V = np.array([2*a]), np.array([-U])
    Ebound = tmm.bound_state_energies(L, V)
    inputwave = np.array([0, 1.0], dtype=complex)
    for n in range(nstates):
        x, psi = tmm.wavefunction(L, V, Ebound[n], inputwave)
        ## Normalize
        normalization = np.trapz(abs(psi)**2, x)
        psi /= np.sqrt(normalization)
        plt.subplot(2, nstates, n+1+nstates)
        plt.plot(x-a, abs(psi)**2)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0,1)
    plt.show()

def resonance_demo_3():
    U, Vb = 6.0, 30.0
    a     = 1.0
    V = np.array([Vb, Vb-U, Vb])

    ## For this E resolution, we can let b range over ~ [1.1, 1.8]
    Evec = np.linspace(24.3, 25.5, 10000)
    bvec = np.linspace(1.05, 2.0, 20)
    T    = np.empty(len(Evec))
    fwhm = np.empty(len(bvec))

    def transmittance(E, dT=0.0):
        k  = np.sqrt(2*E)
        M  = tmm.transfer_matrix(L, V, 0.0, E)
        fm = -M[1,0]/M[1,1]*np.exp(-2j*k*b)
        return np.abs(M[0,0] * np.exp(-2j*k*b) + M[0,1]*fm)**2 - dT

    for ii in range(len(bvec)):
        b = bvec[ii]
        L = np.array([b-a, 2*a, b-a])
        for jj in range(len(Evec)):
            T[jj] = transmittance(Evec[jj])
        ## Locate energies where T = 0.5
        dT  = T - 0.5
        idx = np.nonzero(dT[:-1] * dT[1:] < 0)[0]
        assert len(idx) == 2
        Eres = np.empty(2)
        for n in range(2):
            jj   = idx[n]
            Eres[n] = brentq(transmittance, Evec[jj-1], Evec[jj+2], (0.5,))
        fwhm[ii] = Eres[1] - Eres[0]
        sys.stdout.write('\r' + str(ii+1) + '/' + str(len(bvec)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    plt.plot(bvec-a, fwhm, 'o')

    ## Fermi's Golden Rule prediction
    L, V = np.array([2*a]), np.array([-U])
    E0  = tmm.bound_state_energies(L, V)[0]
    q   = np.sqrt(2*(E0+U))
    et  = np.sqrt(-2*E0)
    k   = np.sqrt(2*(E0+Vb))
    DOS = np.sqrt(2/(E0+Vb))
    Bsq = np.exp(2*et*a)/a/((1+np.sin(2*q*a)/2/q/a)/np.cos(q*a)**2 + 1/et/a)

    bb = np.linspace(1.001, 2.0, 200)

    overlap = Vb * Vb * Bsq / np.pi * np.exp(-2*et*bb) / (k**2 + et**2)
    K = 2*np.pi * overlap * DOS

    plt.plot(bb-a, K)
    plt.xlim(0, 1.0)
    plt.ylim(0, 0.8)
    plt.show()

resonance_demo_3()
