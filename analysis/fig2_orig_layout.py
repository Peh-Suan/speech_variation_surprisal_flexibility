import os, sys, csv, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
from utils.models import replicate_after_length
OUT=f"{PROJ}/analysis/_scratch/fig2"
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"savefig.bbox":"tight",
                     "savefig.pad_inches":0.02})
CMAP="viridis"           # as in the original figure

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
    xt,theta=model.stn(xp,lens,None)
    conv=model.encoder_cnn(xt)[0].numpy()          # (128, T, 17, 17)
    z=model.trunk.encode(model.encoder_proj(
        model.encoder_frame_head(model.encoder_cnn(xt)).permute(0,2,1,3,4)
        .reshape(1,int(lens[0]),-1)), lens)
    recon=model.decode(z,T=x.shape[2],lengths=lens,theta=theta)
T=int(lens[0]); mid=T//2
inp=x[0,0].numpy()                                  # (T, H, W)

# -- choose 4 channels that are not near-duplicates of one another ----------
peak=conv.max((1,2,3)); cand=np.argsort(-peak)[:48]
flat={c: conv[c].ravel() for c in cand}
chosen=[int(cand[0])]
while len(chosen)<4:
    best,bestcor=None,2.0
    for c in cand:
        if c in chosen: continue
        m=max(abs(np.corrcoef(flat[c],flat[s])[0,1]) for s in chosen)
        if m<bestcor: bestcor,best=m,int(c)
    chosen.append(best)
print("channels shown:",[c+1 for c in chosen])

# ---------------------------------------------------------------- panel c --
# Layout follows the original: input frames across the top, a labelled arrow,
# then one row per feature channel and one column per frame.
fig=plt.figure(figsize=(4.4,5.0))
gs=fig.add_gridspec(6,T,height_ratios=[1.30,0.62,1,1,1,1],hspace=0.07,wspace=0.07,
                    top=0.93,bottom=0.015,left=0.11,right=0.99)
vin=(inp.min(),inp.max())
for t in range(T):
    ax=fig.add_subplot(gs[0,t])
    ax.imshow(inp[t],cmap=CMAP,vmin=vin[0],vmax=vin[1],interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_linewidth(0.4)
fig.text((0.11+0.99)/2,0.965,"Time",ha="center",va="center",fontsize=9)

# the arrow sits in its own spacer row, with the label rotated beside it
axa=fig.add_subplot(gs[1,:]); axa.axis("off")
axa.set_xlim(0,1); axa.set_ylim(0,1)
axa.annotate("",xy=(0.5,0.02),xytext=(0.5,0.98),
             arrowprops=dict(arrowstyle="-|>",color="black",lw=1.1,
                             mutation_scale=12))
axa.text(0.448,0.5,"Encoder\n(3D CNN)",rotation=90,ha="center",va="center",fontsize=8.5)

vmax=float(conv[chosen].max())
for r,c in enumerate(chosen):
    for t in range(T):
        ax=fig.add_subplot(gs[r+2,t])
        ax.imshow(conv[c,t],cmap=CMAP,vmin=0,vmax=vmax,interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_linewidth(0.4)
fig.text(0.035,0.33,"Feature",rotation=90,ha="center",va="center",fontsize=9)
fig.savefig(f"{OUT}/fig2c_orig.pdf"); fig.savefig(f"{OUT}/fig2c_orig.png",dpi=600); plt.close(fig)

# ---------------------------------------------------------------- panel d --
imgs=[inp[mid], xt[0,0,mid,:H,:W].numpy(), recon[0,0,mid].numpy()]
lo=min(i.min() for i in imgs); hi=max(i.max() for i in imgs)
fig,axes=plt.subplots(1,3,figsize=(4.8,1.7))
for ax,img in zip(axes,imgs):
    ax.imshow(img,cmap=CMAP,vmin=lo,vmax=hi,interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
fig.subplots_adjust(wspace=0.04,left=0.01,right=0.99,top=0.99,bottom=0.01)
fig.savefig(f"{OUT}/fig2d_orig.pdf"); fig.savefig(f"{OUT}/fig2d_orig.png",dpi=600); plt.close(fig)
print("T=%d mid=%d  conv %s  shared scale [%.2f, %.2f]"%(T,mid+1,conv.shape,lo,hi))
