from skopt import load
from skopt.plots import plot_convergence
import numpy as np
import matplotlib.pyplot as plt
def lwfa_simulation():
	return 0

res = load("../../best optimisation runs/run04_checkpoint.pkl")
x0 = np.array(res.x_iters)
y0 = np.array(res.func_vals)

fig = plt.figure()
ax = plot_convergence(res)
fig = ax.get_figure()
fig.set_size_inches(6,4)
ax.set_ylim(13,16)
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(8,4))
G = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1,1,0.075])
ax = []
a_, w_ = x0[y0.argmin()]
mask = y0 < 1e3
ax.append(fig.add_subplot(G[0]))
img = ax[-1].tripcolor(x0[:,0][mask], 1e6*x0[:,1][mask], y0[mask], 25, vmin=y0[mask].min(), vmax=y0[mask].max())
ax.append(fig.add_subplot(G[1]))
img2 = ax[-1].tricontourf(x0[:,0][mask], 1e6*x0[:,1][mask], y0[mask], 25, vmin=y0[mask].min(), vmax=y0[mask].max())
cax = fig.add_subplot(G[-1])
cbar = fig.colorbar(cax=cax, mappable=img2)
cbar.set_label('Fitness / Relative energy spread (%)')
for AX in ax:
    AX.set_xlabel('$a_0$')
    AX.set_ylabel('$w_0$ ($\mu$m)')
    AX.set_xlim(1.5,5)
    AX.set_ylim(5,15)
    AX.axvline(a_, ls='--', c='lightgrey')
    AX.axhline(1e6*w_, ls='--', c='lightgrey')
    AX.scatter(a_, 1e6*w_, c='lightgrey', zorder=100)
fig.tight_layout()
plt.show()