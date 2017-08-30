from scipy import *
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import sys

## System
psi_up = 0.7
psi_down = sqrt(1.0 - abs(psi_up)**2)
psi_s  = array([psi_up, psi_down], dtype=complex)

## Environment
N = 100
psi_e = randn(N)+1j*randn(N) # Random state
psi_e = psi_e / sqrt(sum(abs(psi_e)**2))

## Interaction Hamiltonian matrix.
A = randn(N,N) + 1j*randn(N,N)
Hs = array([[1.0, 0.0], [0.0,-1.0]]) # Hamiltonian of system
He = (0.5/sqrt(N))*(A + conj(A).T)   # Hamiltonian of environment

## Total Hamiltonian and state
H = kron(Hs, He)
psi = kron(psi_s, psi_e)
## This produces the vector [psi[0]*psi_e, psi[1]*psi_e]

def entropy(psi):
    N = int(len(psi)/2)
    rho = outer(psi, conj(psi))
    rho_e = rho[0:N,0:N] + rho[N:,N:] # Trace over spin subspace
    S = - trace(dot(rho_e, logm(rho_e)))
    return S

def environment_state_overlap(psi):
    N = int(len(psi)/2)
    psi1 = psi[0:N]
    psi2 = psi[N:]
    psi1 = copy(psi1)/sqrt(sum(abs(psi1)**2))
    psi2 = copy(psi2)/sqrt(sum(abs(psi2)**2))
    return abs(dot(conj(psi1), psi2))**2

dt = 0.05
U = expm(-1j*dt*H)

t = arange(0,5.0001,dt)
S = zeros(len(t))
overlap = zeros(len(t))

for n in range(len(t)):
    S[n] = real(entropy(psi))
    overlap[n] = environment_state_overlap(psi)
    psi  = dot(U,psi)

    sys.stdout.write('\r' + str(n+1) + '/' + str(len(t)))
    sys.stdout.flush()
sys.stdout.write('\n')

plt.subplot(2,1,1)
plt.plot(t, S)
plt.xlim(0,5)
plt.ylim(0,1)
plt.subplot(2,1,2)
plt.plot(t, overlap)
plt.xlim(0,5)
plt.ylim(0,1)
plt.show()
