import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility/analysis/_scratch/fig2")
d  = np.load(os.path.expanduser("~/fig2_cd.npz"))
st = np.load(os.path.expanduser("~/fig2_sal_state.npz"))
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.linewidth":0.4,
                     "axes.titlesize":8.5,"savefig.bbox":"tight","savefig.pad_inches":0.02})
n = int(st["done"]); sal = st["acc"]/n
s = float(np.abs(sal).max())
anat = d["original"][int(d["T"])//2]          # same speaker, for anatomical reference
fig, axes = plt.subplots(4, 5, figsize=(6.6, 5.6))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(anat, cmap="gray", interpolation="nearest")
    # saliency on top, transparent where the gradient is near zero, so the
    # anatomy stays visible and the reader can place each hotspot
    a = np.clip(np.abs(sal[k]) / s * 3.0, 0, 1)
    im = ax.imshow(sal[k], cmap="RdBu_r", vmin=-s, vmax=s, alpha=a, interpolation="nearest")
    ax.set_title(f"dim {k+1}", pad=2.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.05, hspace=0.18, right=0.9)
cax = fig.add_axes([0.915, 0.12, 0.013, 0.76])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-s, s))
cb = fig.colorbar(sm, cax=cax); cb.set_label("mean signed gradient", fontsize=7)
cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6, width=0.5)
fig.savefig(f"{OUT}/fig2e.pdf"); fig.savefig(f"{OUT}/fig2e.png", dpi=600); plt.close(fig)
print("re-rendered panel e over anatomy; tokens =", n, " |sal|max = %.3e" % s)
