import os, sys, csv, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
OUT=f"{PROJ}/analysis/_scratch/fig2"
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.linewidth":0.4,
                     "axes.titlesize":8,"savefig.bbox":"tight","savefig.pad_inches":0.02})

BLOB=f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta=json.load(open(BLOB+".meta.json")); H,W=meta["frame_shape"]
mm=np.memmap(BLOB,dtype=np.float16,mode="r").reshape(-1,H,W)
idx={}
with open(BLOB+".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"],round(float(r["start"]),3),round(float(r["end"]),3))]=(int(r["offset"]),int(r["length"]))
TARGET=("/Volumes/Transcend/FR-MRI/P6/OTHER/S1/VIDEO_P6_S1.avi",2.907,2.937)
off,n=idx[TARGET]
frames=np.asarray(mm[off:off+n],np.float32)

spec=CORPORA["frmri"]; model=spec.build_art_model("cpu")
o=torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",map_location="cpu")
model.load_state_dict(o["model"] if isinstance(o,dict) and "model" in o else o); model.eval()
for p in model.parameters(): p.requires_grad_(False)

x,lengths,_=spec.collate()([(torch.from_numpy(frames),0.0)])
T=int(lengths[0]); mid=T//2
anat=x[0,0,mid].numpy().copy()

# ---- panel e, this token only -------------------------------------------
x.requires_grad_(True)
z=model.encode(x,lengths)
D=20; sal=np.zeros((D,H,W),np.float32)
for k in range(D):
    g,=torch.autograd.grad(z[:,k].sum(),x,retain_graph=(k<D-1))
    sal[k]=g[0,0,mid].detach().numpy()
peaks=np.abs(sal).reshape(D,-1).max(1)
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
fig.savefig(f"{OUT}/fig2e_token.pdf"); fig.savefig(f"{OUT}/fig2e_token.png",dpi=600); plt.close(fig)

# ---- panel c, the convolutional feature map ------------------------------
d=np.load(os.path.expanduser("~/fig2_cd.npz"))
fm=d["framehead"][mid]                       # (32, 8, 8), post-ReLU -> non-negative
print("frame-head map: shape",fm.shape," min %.3f max %.3f"%(fm.min(),fm.max()),
      " dead channels:",int((fm.max((1,2))==0).sum()))
vmax=float(fm.max())
fig,axes=plt.subplots(4,8,figsize=(6.6,3.6))
for c,ax in enumerate(axes.ravel()):
    ax.imshow(fm[c],cmap="magma",vmin=0,vmax=vmax,interpolation="nearest")
    ax.set_title(f"{c+1}",pad=2,fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.08,hspace=0.28,right=0.9)
cax=fig.add_axes([0.915,0.15,0.012,0.7])
cb=fig.colorbar(ScalarMappable(cmap="magma",norm=Normalize(0,vmax)),cax=cax)
cb.set_label("activation",fontsize=7)
cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6,width=0.5)
fig.savefig(f"{OUT}/fig2c_featuremap.pdf"); fig.savefig(f"{OUT}/fig2c_featuremap.png",dpi=600); plt.close(fig)
print("single-token saliency peaks (x1e-3):",np.round(peaks*1e3,1).tolist())
