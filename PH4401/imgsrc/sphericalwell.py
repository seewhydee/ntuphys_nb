from scipy import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import spherical_jn, spherical_kn
import sys

## Find and report the energies of a spherical well
def bound_state_energies(a, V0, l):
    E = linspace(-0.9999*V0, -0.00001*V0, 300)
    f = empty(len(E))

    def fun(E):
        q = sqrt(2*(E+V0))
        g = sqrt(-2*E)
        y = q*spherical_jn(l, q*a, True)*spherical_kn(l,g*a) \
          - g*spherical_kn(l, g*a, True)*spherical_jn(l,q*a)
        return log(abs(y))

    for n in range(len(E)):
        q = sqrt(2*(E[n]+V0))
        g = sqrt(-2*E[n])
        f[n] = fun(E[n])
    ## Look for minima of |f|
    idx = nonzero((f[:-2]>f[1:-1]) * (f[1:-1] < f[2:]))[0]
    if len(idx) == 0:
        return []

    Ebound = E[idx + 1]
    ## Refine the guesses
    for n in range(len(Ebound)):
        res = minimize(fun, Ebound[n] - 1e-4)
        Ebound[n] = res.x
    return Ebound


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
            sys.stdout.write('\r' + str(jj+1) + '/' + str(len(V0vec)))
        sys.stdout.flush()
        sys.stdout.write('\n')

        for n in range(nb):
            idx = nonzero(E[n])[0]
            if len(idx) > 0:
                plt.plot(-V0vec[idx], E[n,idx], cols[ll])
    plt.xlim(-V0vec[-1], 0)
    plt.ylim(-20, 0)
    plt.show()


spherical_well_demo()
