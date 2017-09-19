from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import tmm
import sys

## Plot transmittance vs E
def resonance_demo_1(V0, V1, a, b, Emax):
    # Evec = linspace(0.01, Emax, 5000)

    Evec = linspace(10, 12, 5000)

    fm   = empty(len(Evec), dtype=complex)
    fp   = empty(len(Evec), dtype=complex)

    L, V = array([b-a, 2*a, b-a]), array([V1, V0, V1])
    for jj in range(len(Evec)):
        k = sqrt(2*Evec[jj])
        M = tmm.transfer_matrix(L, V, 0.0, Evec[jj])
        fm[jj] = -M[1,0]/M[1,1]*exp(-2j*k*b)
        fp[jj] = M[0,0] * exp(-2j*k*b) + M[0,1]*fm[jj]

    plt.plot(Evec, abs(fp)**2)
    plt.xlim(0, Emax)
    plt.ylim(0,1.05)
    plt.show()

# resonance_demo_1(10.0, 30.0, 1.0, 1.4, 70.0)

def resonance_demo_2():
    from scipy.integrate import trapz
    V0, V1 = 10.0, 30.0
    a, b = 1.0, 1.2
    nstates = 3
    ## From call to Ebound
    Eres = [10.918, 13.646, 18.092, 24.002, 29.973];

    ## Plot resonance wavefunctions (left input = 1)
    L, V = array([b-a, 2*a, b-a]), array([V1, V0, V1])
    for n in range(nstates):
        S = tmm.scattering_matrix(L, V, 0.0, Eres[n])
        components = array([1.0, S[0,0]])
        x, psi = tmm.wavefunction(L, V, Eres[n], components)
        plt.subplot(2, nstates, n+1)
        plt.plot(x-b, abs(psi)**2)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0)

    ## Plot the corresponding square well bound states
    L, V = array([2*a]), array([V0-V1])
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

## Plot transmittance resonance width vs b
def resonance_demo_3():
    V0, V1 =10.0, 30.0
    a = 1.0
    V = array([V1, V0, V1])

    ## For given E resolution, b can range from 1.1 to 1.8
    Evec = linspace(10.6, 11.2, 5000)
    T    = empty(len(Evec))

    bvec = linspace(1.1, 1.6, 20)
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
        ## Find where T = 0.5
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
    plt.plot(bvec, fwhm, 'o')

    ## Fermi's Golden Rule prediction
    L, V = array([2*a]), array([V0-V1])
    E0  = tmm.bound_state_energies(L, V)[0]
    dd  = pi/2 - sqrt(2*(E0+V1-V0))
    et  = sqrt(-2*E0)
    k   = sqrt(2*(E0+V1))
    DOS = sqrt(2/(E0+V1))
    Bsq = exp(2*et*a) * dd*dd/a * pi / (1+pi)

    bb = linspace(1.05, 1.8, 200)
    overlap = (et*cos(k*bb) - k*sin(k*bb))**2 * exp(-2*et*bb) / (2*pi)
    K = 2*pi * Bsq * overlap * DOS

    plt.plot(bb, K)
    plt.xlim(1, 1.6)
    plt.ylim(0, 0.35)
    plt.show()

resonance_demo_3()
