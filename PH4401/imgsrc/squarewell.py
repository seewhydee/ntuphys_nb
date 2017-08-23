from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import trapz

## Calculate transfer matrix across a 1D piecewise-uniform potential.
## L is an array of segment lengths, and k is an array of the
## corresponding wavenumbers (positive or positive-imaginary), both
## denoted from left to right.  kl and kr are the wavenumbers in the
## left and right spaces.
def transfer(L, k, kl, kr):
    assert len(L) == len(k)
    k = append(k, [kr])
    p = kl/k[0]
    M = 0.5*array([[1.+p,1.-p], [1.-p,1.+p]])
    for j in range(len(L)):
        ikL, p = 1j*k[j]*L[j], k[j]/k[j+1]
        M = dot(array([[exp(ikL),0],[0,exp(-ikL)]]), M)
        M = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), M)
    return M

## Given arrays V (segment potentials) and L (segment lengths)
## describing a 1D piecewise-uniform potential, search for the bound
## state energies.  The external potential is assumed to be V=0.
def bound_state_energies(L, V):
    minV = min(V)
    assert minV < 0
    Evec = linspace(0.99999*minV, 0.00001*minV, 1000)

    def M11r(E): # Subroutine: return Re[M[1,1]]
        kout, k = sqrt(2*E), sqrt(2*(E-V))
        M = transfer(L, k, kout, kout)
        return real(M[1,1])
    ## Generate guesses for the bound state energies
    M11vec = empty(len(Evec))
    for n in range(len(Evec)):
        M11vec[n] = M11r(Evec[n])
    ## Use Newton's method to get bound state energies
    idx = nonzero(M11vec[:-1]*M11vec[1:] < 0)
    Ebound = Evec[idx]
    for n in range(len(Ebound)):
        sol = newton(M11r, Ebound[n])
        Ebound[n] = sol
    return Ebound

## Return the wavefunction in the form (x, psi) for a 1D
## piecewise-uniform potential described by length and potential
## arrays L and V, at energy E.  The input components_l specifies the
## wave components at the left edge, while Nx specifies the number of
## x points per segment.  The external potential is assumed to be V=0.
def wavefunction(L, V, E, components_l, Nx=80):
    assert len(L) == len(V) and len(components_l) == 2
    Ltot, x, psi = sum(L), [], []
    components = copy(components_l)

    ## Left side
    x_seg   = linspace(-0.5*Ltot, 0, Nx)
    kout    = sqrt(2*E)
    ikx     = 1j * kout * x_seg
    psi_seg = components[0] * exp(ikx) + components[1] * exp(-ikx)
    x.append(x_seg)
    psi.append(psi_seg)

    ## Loop over interior segments:
    x_curr, k = 0.0, empty(len(V)+1, dtype=complex)
    k[:-1] = sqrt(2*(E-V)); k[-1] = kout
    p = kout/k[0]
    components = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), components)
    for n in range(len(L)):
        ## Calculate and store wavefunction in this segment
        dx = linspace(0, L[n], Nx)
        ikx = 1j * k[n] * dx
        psi_seg = components[0] * exp(ikx) + components[1] * exp(-ikx)
        x.append(x_curr + dx)
        psi.append(psi_seg)
        x_curr += L[n]

        ## Update the wave components using the transfer matrix
        ikL, p = 1j*k[n]*L[n], k[n]/k[n+1]
        components *= array([exp(ikL), exp(-ikL)])
        M = 0.5*array([[1.+p,1.-p], [1.-p,1.+p]])
        components = dot(M, components)

    ## Right side
    dx      = linspace(0, 0.5*Ltot, Nx)
    ikx     = 1j * kout * dx
    psi_seg = components[0] * exp(ikx) + components[1] * exp(-ikx)
    x.append(x_curr + dx)
    psi.append(psi_seg)
    return array(x).flatten(), array(psi).flatten()

## Plot square well bound state wavefunctions
def square_well_demo_1():
    V0, a = 30.0, 1.0 # Square well parameters
    L, V = array([2*a]), array([-V0])
    Ebound = bound_state_energies(L, V)
    print(Ebound)

    ## Plot the bound state wavefunctions
    inputwave = array([0, 1.0], dtype=complex)
    for n in range(len(Ebound)):
        x, psi = wavefunction(L, V, Ebound[n], inputwave)
        ## Normalize
        normalization = trapz(abs(psi)**2, x)
        psi /= sqrt(normalization)
        plt.subplot(len(Ebound),1,n+1)
        plt.plot(x-a, abs(psi)**2)
        plt.xlim(-2*a,2*a)
        plt.ylim(0,1)
    plt.show()

## Plot square well bound state energies vs V0.
def square_well_demo_2():
    a  = 1.0
    L  = array([2*a])
    nb = 10

    V0vec = linspace(0.05, 30.0, 200)
    E = zeros((nb, len(V0vec)))
    for jj in range(len(V0vec)):
        V0 = V0vec[jj]
        V  = array([-V0])
        Eb = sort(bound_state_energies(L, V))
        E[:len(Eb),jj] = Eb

    for n in range(nb):
        idx = nonzero(E[n])[0]
        if len(idx) > 0:
            plt.plot(-V0vec[idx], E[n,idx])
    plt.xlim(-30, 0)
    plt.ylim(-30, 0)
    plt.show()


square_well_demo_2()
