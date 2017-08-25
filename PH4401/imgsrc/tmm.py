## Transfer matrix method for 1D Schrodinger equation.
from scipy import *

## Calculate transfer matrix across a 1D piecewise-uniform potential.
## L is an array of segment lengths, and V is an array of the
## corresponding potentials, both denoted from left to right; Vout is
## the external potential (assumed to be the same on both sides); and
## E is the energy.
def transfer_matrix(L, V, Vout, E):
    assert len(L) == len(V)
    ## Wavenumbers (positive or positive-imaginary)
    kext = sqrt(2*(E-Vout))
    k = append(sqrt(2*(E-V)), [kext])

    p = kext/k[0]
    M = 0.5*array([[1.+p,1.-p], [1.-p,1.+p]], dtype=complex)
    for j in range(len(L)):
        ikL, p = 1j*k[j]*L[j], k[j]/k[j+1]
        M = dot(array([[exp(ikL),0],[0,exp(-ikL)]]), M)
        M = dot(0.5*array([[1.+p,1.-p], [1.-p,1.+p]]), M)
    return M

## Likewise, but for the scattering matrix.
def scattering_matrix(L, V, Vout, E):
    M = transfer_matrix(L, V, Vout, E)
    detM = M[0,0]*M[1,1]-M[1,0]*M[0,1]
    S = array([[-M[1,0],1.0], [detM, M[0,1]]], dtype=complex)/M[1,1]
    return S

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

## Given arrays V (segment potentials) and L (segment lengths)
## describing a 1D piecewise-uniform potential, search for the bound
## state energies.  The external potential is assumed to be V=0.
def bound_state_energies(L, V, nsearch=1000):

    from scipy.optimize import newton

    minV = min(V)
    assert minV < 0
    Evec = linspace(0.99999*minV, 0.00001*minV, nsearch)

    def M11r(E): # Subroutine: return Re[M[1,1]]
        M = transfer_matrix(L, V, 0.0, E)
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
