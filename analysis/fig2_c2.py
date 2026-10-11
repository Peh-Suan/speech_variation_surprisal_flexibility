import os, sys, csv, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
from utils.models import replicate_after_length
OUT=f"{PROJ}/analysis/_scratch/fig2"
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.linewidth":0.4,
                     "savefig.bbox":"tight","savefig.pad_inches":0.02})

BLOB=f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta=json.load(open(BLOB+".meta.json")); H,W=meta["frame_shape"]
mm=np.memmap(BLOB,dtype=np.float16,mode="r").reshape(-1,H,W)
idx={}
with open(BLOB+".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"],round(float(r["start"]),3),round(float(r["end"]),3))]=(int(r["offset"]),int(r["length"]))
off,n=idx[("/Volumes/Transcend/FR-MRI/P6/OTHER/S1/VIDEO_P6_S1.avi",2.907,2.937)]
frames=np.asarray(mm[off:off+n],np.float32)

spec=CORPORA["frmri"]; model=spec.build_art_model("cpu")
o=torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",map_location="cpu")
model.load_state_dict(o["model"] if isinstance(o,dict) and "model" in o else o); model.eval()

x,lengths,_=spec.collate()([(torch.from_numpy(frames),0.0)])
with torch.no_grad():
    lens=model._prepare(x,lengths)
    xp=model._pad_spatial(replicate_after_length(x,lens,time_dim=2))
    xt,_=model.stn(xp,lens,None)
    conv=model.encoder_cnn(xt)            # (1, 128, T, 17, 17) -- the conv stack output
T=int(lens[0]); mid=T//2
fm=conv[0,:,mid].numpy()                  # (128, 17, 17)
alive=(fm.max((1,2))>1e-6).sum()
print("conv-stack map:",fm.shape," active channels: %d/128  max %.2f"%(alive,fm.max()))

order=np.argsort(-fm.max((1,2)))[:32]     # the 32 strongest channels of the 128
vmax=float(fm.max())
fig,axes=plt.subplots(4,8,figsize=(6.6,3.8))
for ax,c in zip(axes.ravel(), order):
    ax.imshow(fm[c],cmap="magma",vmin=0,vmax=vmax,interpolation="nearest")
    ax.set_title(f"ch {c+1}",pad=2,fontsize=6.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.08,hspace=0.30,right=0.9)
cax=fig.add_axes([0.915,0.15,0.012,0.7])
cb=fig.colorbar(ScalarMappable(cmap="magma",norm=Normalize(0,vmax)),cax=cax)
cb.set_label("activation",fontsize=7); cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=6,width=0.5)
fig.savefig(f"{OUT}/fig2c_conv.pdf"); fig.savefig(f"{OUT}/fig2c_conv.png",dpi=600); plt.close(fig)
print("saved fig2c_conv")
