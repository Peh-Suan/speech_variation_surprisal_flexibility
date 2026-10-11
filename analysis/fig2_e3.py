import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
OUT=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility/analysis/_scratch/fig2")
d=np.load(os.path.expanduser("~/fig2_cd.npz")); st=np.load(os.path.expanduser("~/fig2_sal_state.npz"))
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.linewidth":0.4,
                     "axes.titlesize":8,"savefig.bbox":"tight","savefig.pad_inches":0.02})
n=int(st["done"]); sal=st["acc"]/n
anat=d["original"][int(d["T"])//2]
peaks=np.abs(sal).reshape(len(sal),-1).max(1)

def draw(path, per_panel):
    s_all=float(peaks.max())
    fig,axes=plt.subplots(4,5,figsize=(6.6,5.9))
    for k,ax in enumerate(axes.ravel()):
        v = float(peaks[k]) if per_panel else s_all
        ax.imshow(anat, cmap="gray", interpolation="nearest")
        # alpha ramps with the square root of relative magnitude, so mid-level
        # structure is visible instead of only the single hottest blob
        a=np.clip((np.abs(sal[k])/max(v,1e-12))**0.6, 0, 1)
        ax.imshow(sal[k], cmap="RdBu_r", vmin=-v, vmax=v, alpha=a, interpolation="nearest")
        ttl=f"dim {k+1}"
        if per_panel: ttl += f"  $\\pm${peaks[k]*1e3:.0f}"
        ax.set_title(ttl, pad=2.5); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.4)
    fig.subplots_adjust(wspace=0.05, hspace=0.20, right=0.9)
    cax=fig.add_axes([0.915,0.12,0.013,0.76])
    cb=fig.colorbar(ScalarMappable(cmap="RdBu_r", norm=Normalize(-1,1) if per_panel
                                   else Normalize(-s_all,s_all)), cax=cax)
    cb.set_label("signed gradient, relative to each panel's peak" if per_panel
                 else "mean signed gradient", fontsize=7)
    if per_panel:
        cb.set_ticks([-1,0,1]); cb.set_ticklabels(["$-$peak","0","$+$peak"])
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6, width=0.5)
    fig.savefig(path+".pdf"); fig.savefig(path+".png", dpi=600); plt.close(fig)

draw(f"{OUT}/fig2e", True)            # per-panel scaling, peak printed in units of 1e-3
draw(f"{OUT}/fig2e_shared", False)    # one common scale, for comparison
print("per-dim peaks (x1e-3):", np.round(peaks*1e3,1).tolist())
print("ratio strongest/weakest dim: %.0fx" % (peaks.max()/peaks.min()))
