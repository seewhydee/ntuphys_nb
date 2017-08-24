from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import trapz

## Calculate transfer matrix across a 1D piecewise-uniform potential.
## L is an array of segment lengths, and V is an array of the
## corresponding potentials, both denoted from left to right; Vout is
## the external potential (assumed to be the same on both sides); and
## E is the energy.
def transfer(L, V, Vout, E):
    assert len(L) == len(V)
    ## Wavenumbers (positive or positive-imaginary)
    kext = sqrt(2*(E-Vout))
    k = append(sqrt(2*(E-V)), [kext])

    p = kext/k[0]
    M = 0.5*array([[1.+p,1.-p], [1.-p,1.+p]])
    for j in range(len(L)):
        ikL, p = 1j*k[j]*L[j], k[j]/k[j+1]
        M = dot(array([[exp(ikL),0],[0,exp(-ikL)]]), M)
        M = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), M)
    return M

## Plot transmittance vs E
def resonance_demo_1(V0, V1, a, b, Emax):
    Evec = linspace(0.01, Emax, 5000)
    fm   = empty(len(Evec), dtype=complex)
    fp   = empty(len(Evec), dtype=complex)

    L, V = array([b-a, 2*a, b-a]), array([V1, V0, V1])
    for jj in range(len(Evec)):
        k = sqrt(2*Evec[jj])
        M = transfer(L, V, 0.0, Evec[jj])
        fm[jj] = -M[1,0]/M[1,1]*exp(-2j*k*b)
        fp[jj] = M[0,0] * exp(-2j*k*b) + M[0,1]*fm[jj]

    plt.plot(Evec, abs(fp)**2)
    plt.xlim(0, Emax)
    plt.ylim(0,1.05)

resonance_demo_1(10.0, 30.0, 1.0, 1.2, 70.0)
resonance_demo_1(10.0, 30.0, 1.0, 1.4, 70.0)
plt.show()
