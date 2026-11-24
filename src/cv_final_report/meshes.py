"""Reck and Clements decompositions, loss / imperfection simulations,
and a Bloch-Messiah check.  Conventions follow Clements et al., Optica 3, 1460 (2016):

    T_{m,m+1}(theta, phi) acts on modes (m, m+1) as
        [[e^{i phi} cos theta, -sin theta],
         [e^{i phi} sin theta,  cos theta]]
"""
from scipy.stats import unitary_group
import numpy as np

# ---------------------------------------------------------------- elements
def T2(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[np.exp(1j*phi)*c, -s], [np.exp(1j*phi)*s, c]])

def embed(N, m, M2):
    A = np.eye(N, dtype=complex)
    A[m:m+2, m:m+2] = M2
    return A

def null_right(U, r, m):
    """Choose T on (m,m+1) such that (U T^dagger)[r, m] = 0 (mixes columns)."""
    a, b = U[r, m], U[r, m+1]
    if abs(b) < 1e-300:
        theta, phi = np.pi/2, 0.0
    else:
        z = a / b
        theta, phi = np.arctan(abs(z)), np.angle(z)
    return theta, phi

def null_left(U, r, m):
    """Choose T on (m,m+1) such that (T U)[m+1, c] = 0 for column c=r (mixes rows)."""
    a, b = U[m, r], U[m+1, r]
    if abs(a) < 1e-300:
        theta, phi = np.pi/2, 0.0
    else:
        z = -b / a
        theta, phi = np.arctan(abs(z)), np.angle(z)
    return theta, phi

# ---------------------------------------------------------------- Reck
def reck(U):
    """Return D (phases) and ordered list of (m, theta, phi) such that
    U = D * T_K ... T_1, i.e. list[0] is applied first to the light."""
    N = U.shape[0]
    V = U.copy()
    ops = []
    for r in range(N-1, 0, -1):            # null row r, left to right
        for m in range(r):
            th, ph = null_right(V, r, m)
            V = V @ embed(N, m, T2(th, ph)).conj().T
            ops.append((m, th, ph))
    D = np.diag(V)
    # U T1^dag T2^dag ... = D  ->  U = D ... T2 T1 ; T_last applied first
    return D, ops

# ---------------------------------------------------------------- Clements
def clements(U):
    N = U.shape[0]
    V = U.copy()
    right, left = [], []
    for i in range(N-1):
        if i % 2 == 0:
            for j in range(i+1):
                r, m = N-1-j, i-j
                th, ph = null_right(V, r, m)
                V = V @ embed(N, m, T2(th, ph)).conj().T
                right.append((m, th, ph))
        else:
            for j in range(1, i+2):
                r, m = N+j-i-2, N+j-i-3     # null V[r, j-1] with T on rows (m, m+1)
                th, ph = null_left(V, j-1, m)
                V = embed(N, m, T2(th, ph)) @ V
                left.append((m, th, ph))
    D = np.diag(np.diag(V))
    # L_k...L_1 U R_1^dag...R_k^dag = D  ->  U = L_1^dag...L_k^dag D R_k...R_1
    # push each L^dag through D:  T^dag D = D' T'
    for (m, th, ph) in reversed(left):
        M = T2(th, ph).conj().T @ D[m:m+2, m:m+2]
        thp = th
        # M = diag(dm, dn) T(thp, php)
        c, s = np.cos(thp), np.sin(thp)
        if abs(s) > 1e-12 and abs(c) > 1e-12:
            dm, dn = -M[0, 1]/s, M[1, 1]/c
            php = np.angle(M[0, 0]/(dm*c))
        elif abs(s) <= 1e-12:      # identity-like
            dn = M[1, 1]; dm = M[0, 0]; php = 0.0
        else:                      # full swap
            dm, dn = -M[0, 1], M[1, 0]; php = 0.0
            dn = dn  # e^{i php} s = M[1,0]/dn -> php = 0
        D = D.copy(); D[m, m], D[m+1, m+1] = dm, dn
        right.append((m, thp, php))
    # now U = D * R_last ... R_1 ; first applied = right[0]
    return np.diag(D), right

def build(N, D, ops, eta=1.0, bs_err=None):
    """Transfer matrix of mesh. ops applied in order. eta = per-MZI transmission.
    bs_err: list of (e1, e2) coupler errors per op (radians off pi/4)."""
    A = np.eye(N, dtype=complex)
    for k, (m, th, ph) in enumerate(ops):
        if bs_err is None:
            M2 = T2(th, ph)
        else:
            M2 = mzi_physical(th, ph, *bs_err[k])
        A = embed(N, m, np.sqrt(eta)*M2) @ A
    return np.diag(D) @ A

def B(eps):
    a = np.pi/4 + eps
    return np.array([[np.cos(a), 1j*np.sin(a)], [1j*np.sin(a), np.cos(a)]])

def mzi_physical(th, ph, e1, e2):
    """MZI from two directional couplers; equals T2(th,ph) when e1=e2=0."""
    alpha = np.pi - 2*th
    g = -1j*np.exp(-1j*alpha/2)
    P = np.diag([np.exp(1j*alpha), 1])
    return g * B(e2) @ P @ B(e1) @ np.diag([np.exp(1j*ph), -1])

def fidelity(U, A):
    N = U.shape[0]
    return abs(np.trace(U.conj().T @ A))**2 / (N*np.trace(A.conj().T @ A).real)

def depth(N, ops):
    layer = np.zeros(N, int)
    for (m, *_ ) in ops:
        d = max(layer[m], layer[m+1]) + 1
        layer[m] = layer[m+1] = d
    return layer.max()

# ---------------------------------------------------------------- Bloch-Messiah
def random_symplectic_complex(N, rng):
    U = unitary_group.rvs(N, random_state=rng); V = unitary_group.rvs(N, random_state=rng)
    r = rng.uniform(0.1, 1.2, N)
    A = U @ np.diag(np.cosh(r)) @ V.conj().T
    Bm = U @ np.diag(np.sinh(r)) @ V.T
    return A, Bm

def bloch_messiah(A, Bm):
    U, c, Vh = np.linalg.svd(A)
    V = Vh.conj().T
    r = np.arccosh(c)
    S = U.conj().T @ Bm @ V.conj()          # = diag(sinh r) * Phi^2
    ph = np.angle(np.diag(S))
    Phi = np.diag(np.exp(1j*ph/2))
    return U @ Phi, r, V @ Phi
