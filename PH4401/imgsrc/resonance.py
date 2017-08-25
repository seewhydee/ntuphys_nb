from scipy import *
import matplotlib.pyplot as plt
import tmm

## Plot transmittance vs E
def resonance_demo_1(V0, V1, a, b, Emax):
    Evec = linspace(0.01, Emax, 5000)
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

resonance_demo_2()

