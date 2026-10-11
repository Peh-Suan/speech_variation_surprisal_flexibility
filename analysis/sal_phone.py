import os, sys, csv, json, time
import numpy as np, torch
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.corpora import CORPORA
PHONE=sys.argv[1]; BUDGET=float(sys.argv[2]) if len(sys.argv)>2 else 130.0
SPK="/P6/"; D=20; BATCH=8
STATE=os.path.expanduser(f"~/sal_{PHONE}.npz")

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
        if SPK not in r["art_f"]: continue
        for p in ("phone1","phone2"):
            if r[p]==PHONE:
                k=(r["art_f"],round(float(r[f"{p}_start"]),3),round(float(r[f"{p}_end"]),3))
                if k in idx and k not in toks: toks.append(k)
toks.sort()
if os.environ.get("COUNT_ONLY"):
    print(f"/{PHONE}/ tokens for P6 in cache: {len(toks)}"); sys.exit()

spec=CORPORA["frmri"]; model=spec.build_art_model("cpu")
o=torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",map_location="cpu")
model.load_state_dict(o["model"] if isinstance(o,dict) and "model" in o else o); model.eval()
for p in model.parameters(): p.requires_grad_(False)

if os.path.exists(STATE):
    st=np.load(STATE); acc,anat,done=st["acc"],st["anat"],int(st["done"])
else:
    acc,anat,done=np.zeros((D,H,W)),np.zeros((H,W)),0

t0=time.time()
while done<len(toks) and time.time()-t0<BUDGET:
    batch=toks[done:done+BATCH]
    arrs=[torch.from_numpy(np.asarray(mm[idx[k][0]:idx[k][0]+idx[k][1]],np.float32)) for k in batch]
    x,lengths,_=spec.collate()([(a,0.0) for a in arrs]); x.requires_grad_(True)
    z=model.encode(x,lengths)
    for b in range(len(batch)):
        anat+=x[b,0,int(lengths[b])//2].detach().numpy()      # mean posture, for the underlay
    for k in range(D):
        g,=torch.autograd.grad(z[:,k].sum(),x,retain_graph=(k<D-1))
        for b in range(len(batch)):
            acc[k]+=g[b,0,int(lengths[b])//2].detach().numpy()
    done+=len(batch)
np.savez(STATE,acc=acc,anat=anat,done=done,total=len(toks))
print(f"/{PHONE}/: {done}/{len(toks)} tokens ({time.time()-t0:.0f}s)")
