import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import sys

## System
psi_up = 0.7
psi_down = np.sqrt(1.0 - abs(psi_up)**2)
psi_s  = np.array([psi_up, psi_down], dtype=complex)

## Environment
N = 100
psi_e = np.random.randn(N)+1j*np.random.randn(N) # Random state
psi_e = psi_e / np.sqrt(np.sum(abs(psi_e)**2))

## Interaction Hamiltonian matrix.
A = np.random.randn(N,N) + 1j*np.random.randn(N,N)
Hs = np.array([[1.0, 0.0], [0.0,-1.0]]) # Hamiltonian of system
He = (0.5/np.sqrt(N))*(A + np.conj(A).T)   # Hamiltonian of environment

## Total Hamiltonian and state
H = np.kron(Hs, He)
psi = np.kron(psi_s, psi_e)
## This produces the vector [psi[0]*psi_e, psi[1]*psi_e]

def entropy(psi):
    N = int(len(psi)/2)
    rho = np.outer(psi, np.conj(psi))
    rho_e = rho[0:N,0:N] + rho[N:,N:] # Trace over spin subspace
    S = - np.trace(np.dot(rho_e, logm(rho_e)))
    return S

def environment_state_overlap(psi):
    N = int(len(psi)/2)
    psi1 = psi[0:N]
    psi2 = psi[N:]
    psi1 = np.copy(psi1)/np.sqrt(np.sum(abs(psi1)**2))
    psi2 = np.copy(psi2)/np.sqrt(np.sum(abs(psi2)**2))
    return abs(np.dot(np.conj(psi1), psi2))**2

dt = 0.05
U = expm(-1j*dt*H)

t = np.arange(0,4.0001,dt)
S = np.zeros(len(t))
overlap = np.zeros(len(t))

for n in range(len(t)):
    S[n] = np.real(entropy(psi))
    overlap[n] = environment_state_overlap(psi)
    psi  = np.dot(U,psi)

    sys.stdout.write('\r' + str(n+1) + '/' + str(len(t)))
    sys.stdout.flush()
sys.stdout.write('\n')

plt.subplot(2,1,1)
plt.plot(t, S)
plt.xlim(0,4)
plt.ylim(0,1)

plt.subplot(2,1,2)
plt.plot(t, overlap)
## Plot predicted entropy for comparison
p1, p2 = abs(psi_up)**2, abs(psi_down)**2
S = - p1*np.log(p1) - p2*np.log(p2)
plt.plot([0,4], [S,S])
plt.xlim(0,4)
plt.ylim(0,1)
plt.show()
