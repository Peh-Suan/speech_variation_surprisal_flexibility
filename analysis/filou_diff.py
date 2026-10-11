import os, sys, csv, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
OUT=f"{PROJ}/analysis/_scratch/fig2"

BLOB=f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta=json.load(open(BLOB+".meta.json")); H,W=meta["frame_shape"]
mm=np.memmap(BLOB,dtype=np.float16,mode="r").reshape(-1,H,W)
idx={}
with open(BLOB+".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"],round(float(r["start"]),3),round(float(r["end"]),3))]=(int(r["offset"]),int(r["length"]))
REC="/Volumes/Transcend/FR-MRI/P6/OTHER/S1/VIDEO_P6_S1.avi"
TOK={"l":(REC,2.907,2.937),"u":(REC,2.937,3.067)}   # the /l/ and the /u/ of filou

spec=CORPORA["frmri"]; model=spec.build_art_model("cpu")
o=torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",map_location="cpu")
model.load_state_dict(o["model"] if isinstance(o,dict) and "model" in o else o); model.eval()
for p in model.parameters(): p.requires_grad_(False)

D=20; res={}
for ph,key in TOK.items():
    off,n=idx[key]
    fr=np.asarray(mm[off:off+n],np.float32)
    x,lengths,_=spec.collate()([(torch.from_numpy(fr),0.0)])
    T=int(lengths[0]); mid=T//2
    anat=x[0,0,mid].numpy().copy()
    x.requires_grad_(True)
    z=model.encode(x,lengths)
    sal=np.zeros((D,H,W),np.float32)
    for k in range(D):
        g,=torch.autograd.grad(z[:,k].sum(),x,retain_graph=(k<D-1))
        sal[k]=g[0,0,mid].detach().numpy()
    res[ph]=dict(sal=sal,anat=anat,T=T)
    print(f"/{ph}/ frames={T} mid={mid+1} peak|grad|x1e-3 median={np.median(np.abs(sal).reshape(D,-1).max(1))*1e3:.1f}")

L,U=res["l"]["sal"],res["u"]["sal"]; diff=L-U
cors=[float(np.corrcoef(L[k].ravel(),U[k].ravel())[0,1]) for k in range(D)]
rel=np.abs(diff).reshape(D,-1).max(1)/np.abs(L).reshape(D,-1).max(1)
print("\ncorr(/l/,/u/) per dim: median %.3f  min %.3f  max %.3f"%(np.median(cors),min(cors),max(cors)))
print("|diff|max / |l|max  : median %.2f  min %.2f  max %.2f"%(np.median(rel),rel.min(),rel.max()))
print("anat corr between the two middle frames: %.3f"%
      np.corrcoef(res["l"]["anat"].ravel(),res["u"]["anat"].ravel())[0,1])

plt.rcParams.update({"font.family":"sans-serif","font.size":9,
                     "savefig.bbox":"tight","savefig.pad_inches":0.03})
def draw(S,anat,path):
    fig,axes=plt.subplots(4,5,figsize=(4.7,4.9))
    for k,ax in enumerate(axes.ravel()):
        v=float(np.abs(S[k]).max())
        ax.imshow(anat,cmap="gray",interpolation="bilinear")
        ax.imshow(S[k],cmap="bwr",vmin=-v,vmax=v,alpha=0.62,interpolation="bilinear")
        ax.set_title(str(k+1),pad=2.0,fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
    fig.suptitle("Latent dimension",y=0.985,fontsize=10)
    fig.subplots_adjust(wspace=0.06,hspace=0.14,top=0.925,bottom=0.01,left=0.01,right=0.99)
    fig.savefig(path+".pdf"); fig.savefig(path+".png",dpi=600); plt.close(fig)

draw(diff,res["l"]["anat"],f"{OUT}/fig2e_filou_diff")
draw(L,   res["l"]["anat"],f"{OUT}/fig2e_filou_l")
draw(U,   res["u"]["anat"],f"{OUT}/fig2e_filou_u")
np.savez(os.path.expanduser("~/filou_sal.npz"),L=L,U=U,
         anat_l=res["l"]["anat"],anat_u=res["u"]["anat"],rel=rel,cors=np.array(cors))
print("saved")
