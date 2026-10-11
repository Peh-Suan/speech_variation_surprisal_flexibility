import os, sys, csv, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
OUT=f"{PROJ}/analysis/_scratch/fig2"
st=np.load(os.path.expanduser("~/sal_u.npz"))
n=int(st["done"]); sal=st["acc"]/n; meanframe=st["anat"]/n

# a single /u/ token's middle frame as the underlay: averaging 143 postures
# blurs the tract outline, and the old panel's anatomy is crisp
BLOB=f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta=json.load(open(BLOB+".meta.json")); H,W=meta["frame_shape"]
mm=np.memmap(BLOB,dtype=np.float16,mode="r").reshape(-1,H,W)
idx={}
with open(BLOB+".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"],round(float(r["start"]),3),round(float(r["end"]),3))]=(int(r["offset"]),int(r["length"]))
toks=[]
with open(f"{PROJ}/phone_based/data_processing/dataset_construction/datasets/processed_frmri.csv") as fh:
    for r in csv.DictReader(fh):
        if "/P6/" not in r["art_f"]: continue
        for p in ("phone1","phone2"):
            if r[p]=="u":
                k=(r["art_f"],round(float(r[f"{p}_start"]),3),round(float(r[f"{p}_end"]),3))
                if k in idx and k not in toks: toks.append(k)
toks.sort()
lens=np.array([idx[k][1] for k in toks])
pick=toks[int(np.argsort(np.abs(lens-np.median(lens)))[0])]     # median-length token
o,l=idx[pick]
crisp=np.asarray(mm[o:o+l],np.float32)[l//2]
print("underlay token:",pick," frames:",l)

plt.rcParams.update({"font.family":"sans-serif","font.size":9,
                     "savefig.bbox":"tight","savefig.pad_inches":0.03})
def draw(anat, path):
    fig,axes=plt.subplots(4,5,figsize=(4.7,4.9))
    for k,ax in enumerate(axes.ravel()):
        v=float(np.abs(sal[k]).max())
        ax.imshow(anat,cmap="gray",interpolation="bilinear")
        ax.imshow(sal[k],cmap="bwr",vmin=-v,vmax=v,alpha=0.62,interpolation="bilinear")
        ax.set_title(str(k+1),pad=2.0,fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
    fig.suptitle("Latent dimension",y=0.985,fontsize=10)
    fig.subplots_adjust(wspace=0.06,hspace=0.14,top=0.925,bottom=0.01,left=0.01,right=0.99)
    fig.savefig(path+".pdf"); fig.savefig(path+".png",dpi=600); plt.close(fig)

draw(crisp,     f"{OUT}/fig2e_u")
draw(meanframe, f"{OUT}/fig2e_u_meanframe")
print("done")
