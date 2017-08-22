from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import newton

## Calculate transfer matrix across a 1D piecewise-uniform potential.
## L is an array of segment lengths, and k is an array of the
## corresponding wavenumbers (positive or positive-imaginary), both
## denoted from left to right.  kl and kr are the wavenumbers in the
## left and right spaces.
def transfer(L, k, kl, kr):
    assert len(L) == len(k)
    p = k[0]/kl
    M = 0.5*array([[1.+p,1.-p], [1.-p,1.+p]])
    for j in range(len(L)-1):
        kL, p = k[j]*L[j], k[j+1]/k[j]
        M = dot(array([[exp(1j*kL),0],[0,exp(-1j*kL)]]), M)
        M = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), M)
    kL, p = k[-1]*L[-1], kr/k[-1]
    M = dot(array([[exp(1j*kL),0],[0,exp(-1j*kL)]]), M)
    M = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), M)
    return M

def square_well_bound_states(V0, a):
    assert V0 > 2e-6
    V, L = array([-V0]), array([2*a])
    Evec = linspace(-V0+1e-6,-1e-6,1000)

    def M11r(E): # Subroutine: return Re[M[1,1]]
        kout, k = sqrt(2*E), sqrt(2*(E-V))
        M = transfer(L, k, kout, kout)
        return real(M[1,1])

    ## Generate guesses for the bound state energies
    M11vec = zeros(len(Evec))
    for n in range(len(Evec)):
        M11vec[n] = M11r(Evec[n])
    ## Use Newton's method to get bound state energies
    idx = nonzero(M11vec[:-1]*M11vec[1:] < 0)
    Ebound = Evec[idx]
    for n in range(len(Ebound)):
        sol = newton(M11r, Ebound[n])
        Ebound[n] = sol

    plt.plot(Evec, real(M11vec), Evec, abs(M11vec), Evec, imag(M11vec))
    plt.plot(Ebound, zeros(len(Ebound)), 'bo')
    plt.ylim(-10,10)
    plt.show()

square_well_bound_states(30.0, 1.0)
