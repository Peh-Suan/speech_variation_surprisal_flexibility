import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility/analysis/_scratch/fig2")
d   = np.load(os.path.expanduser("~/fig2_cd.npz"))
st  = np.load(os.path.expanduser("~/fig2_sal_state.npz"))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# signed quantities get a diverging map with a neutral midpoint and symmetric
# limits, so zero is always the neutral colour and sign is readable
DIV = "RdBu_r"

T = int(d["T"])
mid = T // 2

# ---------------------------------------------------------------- panel c --
emb = d["embedding"]                                   # (T, 256)
v = float(np.abs(emb).max())
fig, ax = plt.subplots(figsize=(6.4, 1.5))
im = ax.imshow(emb, aspect="auto", cmap=DIV, vmin=-v, vmax=v, interpolation="nearest")
ax.set_xlabel("frame-embedding dimension")
ax.set_ylabel("frame")
ax.set_yticks(range(T), [str(i + 1) for i in range(T)])
ax.set_xticks([0, 63, 127, 191, 255], ["1", "64", "128", "192", "256"])
for s in ax.spines.values():
    s.set_linewidth(0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=6, width=0.5)
fig.savefig(f"{OUT}/fig2c.pdf"); fig.savefig(f"{OUT}/fig2c.png", dpi=600); plt.close(fig)

# ---------------------------------------------------------------- panel d --
orig, tr, rec = d["original"][mid], d["transformed"][mid], d["recon"][mid]
lo = min(orig.min(), tr.min(), rec.min()); hi = max(orig.max(), tr.max(), rec.max())
fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
for ax, img, t in zip(axes, (orig, tr, rec),
                      ("original", "after spatial transform", "reconstruction")):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
fig.subplots_adjust(wspace=0.06)
fig.savefig(f"{OUT}/fig2d.pdf"); fig.savefig(f"{OUT}/fig2d.png", dpi=600); plt.close(fig)

# difference, kept separate in case it is wanted
dif = rec - orig; w = float(np.abs(dif).max())
fig, ax = plt.subplots(figsize=(2.0, 2.0))
im = ax.imshow(dif, cmap=DIV, vmin=-w, vmax=w, interpolation="nearest")
ax.set_title("reconstruction $-$ original"); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6, width=0.5)
fig.savefig(f"{OUT}/fig2d_diff.pdf"); fig.savefig(f"{OUT}/fig2d_diff.png", dpi=600); plt.close(fig)

# ---------------------------------------------------------------- panel e --
n_tok = int(st["done"])
sal = st["acc"] / n_tok                                 # (20, H, W) mean gradient
s = float(np.abs(sal).max())
fig, axes = plt.subplots(4, 5, figsize=(6.6, 5.6))
for k, ax in enumerate(axes.ravel()):
    im = ax.imshow(sal[k], cmap=DIV, vmin=-s, vmax=s, interpolation="nearest")
    ax.set_title(f"dim {k+1}", pad=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.05, hspace=0.18, right=0.9)
cax = fig.add_axes([0.915, 0.12, 0.013, 0.76])
cb = fig.colorbar(im, cax=cax)
cb.set_label("mean signed gradient", fontsize=7)
cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6, width=0.5)
fig.savefig(f"{OUT}/fig2e.pdf"); fig.savefig(f"{OUT}/fig2e.png", dpi=600); plt.close(fig)

print("T=%d mid-frame=%d  tokens averaged for e: %d" % (T, mid + 1, n_tok))
print("emb |max| %.3f   saliency |max| %.2e   recon-orig |max| %.3f" % (v, s, w))
print("files:", sorted(os.listdir(OUT)))
