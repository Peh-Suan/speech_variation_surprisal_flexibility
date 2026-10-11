import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
OUT=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility/analysis/_scratch/fig2")
st=np.load(os.path.expanduser("~/sal_u.npz"))
n=int(st["done"]); sal=st["acc"]/n; anat=st["anat"]/n
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.linewidth":0.4,
                     "axes.titlesize":8,"savefig.bbox":"tight","savefig.pad_inches":0.02})
peaks=np.abs(sal).reshape(len(sal),-1).max(1)
fig,axes=plt.subplots(4,5,figsize=(6.6,5.9))
for k,ax in enumerate(axes.ravel()):
    v=float(peaks[k])
    ax.imshow(anat,cmap="gray",interpolation="nearest")
    a=np.clip((np.abs(sal[k])/max(v,1e-12))**0.6,0,1)
    ax.imshow(sal[k],cmap="RdBu_r",vmin=-v,vmax=v,alpha=a,interpolation="nearest")
    ax.set_title(f"dim {k+1}  $\\pm${v*1e3:.0f}",pad=2.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.05,hspace=0.20,right=0.9)
cax=fig.add_axes([0.915,0.12,0.013,0.76])
cb=fig.colorbar(ScalarMappable(cmap="RdBu_r",norm=Normalize(-1,1)),cax=cax)
cb.set_label("signed gradient, relative to each panel's peak",fontsize=7)
cb.set_ticks([-1,0,1]); cb.set_ticklabels(["$-$peak","0","$+$peak"])
cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6,width=0.5)
fig.savefig(f"{OUT}/fig2e_u.pdf"); fig.savefig(f"{OUT}/fig2e_u.png",dpi=600); plt.close(fig)

print("/u/ tokens:",n)
print("peaks x1e-3:",np.round(peaks*1e3,1).tolist())
# how much does the map differ from the /l/ one?
l=np.load(os.path.expanduser("~/fig2_sal_state.npz")); sl=l["acc"]/int(l["done"])
cors=[float(np.corrcoef(sal[k].ravel(), sl[k].ravel())[0,1]) for k in range(len(sal))]
print("per-dim corr(/u/ map, /l/ map): median %.3f  min %.3f  max %.3f" %
      (np.median(cors), min(cors), max(cors)))
print("peak ratio /u/ over /l/:", np.round(peaks/ (np.abs(sl).reshape(20,-1).max(1)),2).tolist())
