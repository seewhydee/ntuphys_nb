from scipy import *
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import tmm

## Plot square well bound state wavefunctions
def square_well_demo_1():
    V0, a = 20.0, 1.0 # Square well parameters
    L, V = array([2*a]), array([-V0])
    Ebound = tmm.bound_state_energies(L, V)
    print(Ebound)

    ## Plot the bound state wavefunctions
    inputwave = array([0, 1.0], dtype=complex)
    for n in range(len(Ebound)):
        x, psi = tmm.wavefunction(L, V, Ebound[n], inputwave)
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
        Eb = sort(tmm.bound_state_energies(L, V))
        E[:len(Eb),jj] = Eb

    for n in range(nb):
        idx = nonzero(E[n])[0]
        if len(idx) > 0:
            plt.plot(-V0vec[idx], E[n,idx])
    plt.xlim(-30, 0)
    plt.ylim(-30, 0)
    plt.show()

square_well_demo_1()
square_well_demo_2()
