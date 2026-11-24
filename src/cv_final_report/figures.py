import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from meshes import *

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
C_RECK, C_CLEM = "#B0413E", "#2F5D8A"
OUT = "/home/claude/report/figs/"
rng = np.random.default_rng(2026)
results = {}

# ------------------------------------------------------------ Fig 1: mesh layouts
def draw_mesh(ax, N, ops, title, order_labels):
    # place each MZI at its depth layer
    layer = np.zeros(N, int); pos = []
    for (m, *_ ) in ops:
        d = max(layer[m], layer[m+1]) + 1
        layer[m] = layer[m+1] = d; pos.append(d)
    L = max(pos)
    for k in range(N):
        ax.plot([0.3, L+0.7], [k, k], color="0.35", lw=0.8, zorder=0)
        ax.text(0.15, k, f"{k+1}", ha="right", va="center", fontsize=7)
    for idx, ((m, *_), d) in enumerate(zip(ops, pos)):
        ax.add_patch(plt.Rectangle((d-0.32, m-0.12), 0.64, 1.24, fc="white",
                                   ec=order_labels[idx][1], lw=1.1, zorder=2))
        ax.text(d, m+0.5, order_labels[idx][0], ha="center", va="center", fontsize=6.5, zorder=3)
    ax.set_xlim(-0.2, L+1); ax.set_ylim(N-0.4, -0.6); ax.axis("off")
    ax.set_title(title, fontsize=9)

N = 6
U = unitary_group.rvs(N, random_state=rng)
_, ops_r = reck(U)
labs_r = [(str(i+1), C_RECK) for i in range(len(ops_r))]
_, ops_c = clements(U)
# label Clements MZIs by the diagonal (nulling step) that fixed them
nR = sum(range(1, N, 2))  # number nulled from the right = MZIs in even i
diag_of = []
for i in range(N-1):
    if i % 2 == 0: diag_of += [i+1]*(i+1)
left_diags = []
for i in range(N-1):
    if i % 2 == 1: left_diags += [i+1]*(i+1)
diag_of += left_diags[::-1]
labs_c = [(f"d{d}", C_CLEM if d % 2 else "#6A8F3C") for d in diag_of]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [2*N-3, N]})
draw_mesh(axs[0], N, ops_r, f"(a) Reck: depth $2N-3={2*N-3}$", labs_r)
draw_mesh(axs[1], N, ops_c, f"(b) Clements: depth $N={N}$", labs_c)
plt.tight_layout(); plt.savefig(OUT+"meshes.pdf"); plt.close()

# ------------------------------------------------------------ Fig 2: loss tolerance
def dB2eta(dB): return 10**(-dB/10)
Ns = list(range(3, 21))
nsamp = 300
FR, FC = [], []
for N in Ns:
    fr, fc = [], []
    for _ in range(nsamp):
        U = unitary_group.rvs(N, random_state=rng)
        D, o = reck(U);     fr.append(fidelity(U, build(N, D, o, eta=dB2eta(0.2))))
        D, o = clements(U); fc.append(fidelity(U, build(N, D, o, eta=dB2eta(0.2))))
    FR.append((np.mean(fr), np.std(fr))); FC.append((np.mean(fc), np.std(fc)))
FR, FC = np.array(FR), np.array(FC)
losses = np.linspace(0, 1.0, 11)
GR, GC = [], []
N = 20
Us = [unitary_group.rvs(N, random_state=rng) for _ in range(100)]
dec_r = [reck(U) for U in Us]; dec_c = [clements(U) for U in Us]
for l in losses:
    GR.append(np.mean([fidelity(U, build(N, *d, eta=dB2eta(l))) for U, d in zip(Us, dec_r)]))
    GC.append(np.mean([fidelity(U, build(N, *d, eta=dB2eta(l))) for U, d in zip(Us, dec_c)]))
results["loss"] = {"N": Ns, "reck": FR[:, 0].tolist(), "clem": FC[:, 0].tolist(),
                   "losses": losses.tolist(), "reck20": GR, "clem20": GC}

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax = axs[0]
ax.errorbar(Ns, FR[:, 0], FR[:, 1], color=C_RECK, marker="o", ms=3, lw=1, capsize=0, label="Reck")
ax.errorbar(Ns, FC[:, 0], FC[:, 1], color=C_CLEM, marker="s", ms=3, lw=1, capsize=0, label="Clements")
ax.set_xlabel("number of modes $N$"); ax.set_ylabel(r"fidelity $\mathcal{F}$"); ax.legend(frameon=False)
ax.set_title("(a) 0.2 dB loss per MZI", fontsize=9)
ax = axs[1]
ax.plot(losses, GR, color=C_RECK, marker="o", ms=3, lw=1, label="Reck")
ax.plot(losses, GC, color=C_CLEM, marker="s", ms=3, lw=1, label="Clements")
ax.set_xlabel("loss per MZI (dB)"); ax.set_ylabel(r"fidelity $\mathcal{F}$")
ax.set_title("(b) $N=20$", fontsize=9)
plt.tight_layout(); plt.savefig(OUT+"loss.pdf"); plt.close()

# ------------------------------------------------------------ Fig 3: Haar statistics + coupler errors
N = 12
nsamp = 2000
ops_ref = None
S = None
for _ in range(nsamp):
    U = unitary_group.rvs(N, random_state=rng)
    _, o = clements(U)
    th = np.array([x[1] for x in o])
    S = np.sin(th)**2 if S is None else S + np.sin(th)**2
    if ops_ref is None: ops_ref = o
S /= nsamp
layer = np.zeros(N, int); grid = np.full((N-1, N), np.nan)
for (m, *_ ), s in zip(ops_ref, S):
    d = max(layer[m], layer[m+1]) + 1; layer[m] = layer[m+1] = d
    grid[m, d-1] = s
results["haar_grid_min"] = float(np.nanmin(grid)); results["haar_grid_max"] = float(np.nanmax(grid))

sig = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
ER, EC = [], []
Nm = 12
Us = [unitary_group.rvs(Nm, random_state=rng) for _ in range(150)]
dec_r = [reck(U) for U in Us]; dec_c = [clements(U) for U in Us]
for s in sig:
    er, ec = [], []
    for U, dr, dc in zip(Us, dec_r, dec_c):
        err = rng.normal(0, s, (len(dr[1]), 2))
        er.append(1-fidelity(U, build(Nm, *dr, bs_err=err)))
        err = rng.normal(0, s, (len(dc[1]), 2))
        ec.append(1-fidelity(U, build(Nm, *dc, bs_err=err)))
    ER.append(np.mean(er)); EC.append(np.mean(ec))
results["bs"] = {"sigma": sig.tolist(), "reck": ER, "clem": EC}

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
im = ax.imshow(grid, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xlabel("column (depth layer)"); ax.set_ylabel("upper mode $m$ of MZI")
ax.set_xticks(range(N)); ax.set_xticklabels(range(1, N+1), fontsize=7)
ax.set_yticks(range(N-1)); ax.set_yticklabels(range(1, N), fontsize=7)
cb = plt.colorbar(im, ax=ax, fraction=0.05); cb.set_label(r"$\langle\sin^2\theta\rangle$")
ax.set_title(f"(a) Haar-averaged mixing, $N={N}$", fontsize=9)
ax = axs[1]
ax.semilogy(sig[1:], np.array(ER[1:]), color=C_RECK, marker="o", ms=3, lw=1, label="Reck")
ax.semilogy(sig[1:], np.array(EC[1:]), color=C_CLEM, marker="s", ms=3, lw=1, label="Clements")
ax.set_xlabel(r"coupler error std. $\sigma_\epsilon$ (rad)"); ax.set_ylabel(r"infidelity $1-\mathcal{F}$")
ax.legend(frameon=False); ax.set_title(f"(b) imperfect couplers, $N={Nm}$", fontsize=9)
plt.tight_layout(); plt.savefig(OUT+"errors.pdf"); plt.close()

# ------------------------------------------------------------ Bloch-Messiah numerical check
errs = []
for _ in range(50):
    A, Bm = random_symplectic_complex(8, rng)
    Uu, r, V = bloch_messiah(A, Bm)
    # synthesise both passive parts with Clements meshes
    D1, o1 = clements(Uu); D2, o2 = clements(V.conj().T)
    Uu2 = build(8, D1, o1); Vh2 = build(8, D2, o2)
    A2 = Uu2 @ np.diag(np.cosh(r)) @ Vh2
    B2 = Uu2 @ np.diag(np.sinh(r)) @ Vh2.conj()
    errs.append(max(np.abs(A2-A).max(), np.abs(B2-Bm).max()))
results["bloch_messiah_maxerr"] = float(max(errs))
print(json.dumps(results, indent=1))
json.dump(results, open("/home/claude/code/results.json", "w"), indent=1)
