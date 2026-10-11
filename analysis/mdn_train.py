import os, sys, time, copy, pickle
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.models import MDN
STATE=os.path.expanduser("~/mdn_state.pt")
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 140.0
torch.manual_seed(0); np.random.seed(0)

d=np.load(f"{PROJ}/phone_based/flexibility_estimation/latents/aemri75anno__prop25.npz",allow_pickle=True)
keep=d["has_art"]&d["has_ac"]
art=d["art"][keep].astype(np.float32); ac=d["ac"][keep].astype(np.float32)
lab=d["labels"][keep]; sub=d["subjects"][keep]
N=len(art)

# the token the figure illustrates: a /k/ by sub013. Choose the one whose
# predicted mixture is most spread out -- decided after training, so here we
# only hold out every sub013 /k/ so none of them is in the training set.
held=np.where((lab=="k")&(sub=="sub013"))[0]
rest=np.setdiff1d(np.arange(N),held)
rng=np.random.RandomState(0); rng.shuffle(rest)
split=int(0.9*len(rest)); tr,va=rest[:split],rest[split:]

sc_ac,sc_art=StandardScaler().fit(ac[tr]),StandardScaler().fit(art[tr])
acn,artn=sc_ac.transform(ac).astype(np.float32),sc_art.transform(art).astype(np.float32)

class AX(Dataset):
    def __init__(s,X,Y): s.X=torch.from_numpy(X); s.Y=torch.from_numpy(Y)
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return s.X[i],s.Y[i]
ds=AX(acn,artn)
tl=DataLoader(Subset(ds,tr),batch_size=64,shuffle=True)
vl=DataLoader(Subset(ds,va),batch_size=256,shuffle=False)

model=MDN(x_dim=64,y_dim=64,n_components=5,hidden_dim=256)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
best,bstate,counter,epoch=float("inf"),None,0,0
if os.path.exists(STATE):
    s=torch.load(STATE,map_location="cpu")
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
    best,bstate,counter,epoch=s["best"],s["bstate"],s["counter"],s["epoch"]

t0=time.time(); stop=False
while time.time()-t0<BUDGET and counter<10 and epoch<300:
    epoch+=1; model.train()
    for Xb,Yb in tl:
        loss=-model.log_prob(Xb,Yb).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vs=0.0
    with torch.no_grad():
        for Xb,Yb in vl: vs+=-model.log_prob(Xb,Yb).sum().item()
    vs/=len(va)
    if vs<best-1e-4: best,bstate,counter=vs,copy.deepcopy(model.state_dict()),0
    else: counter+=1
    print(f"epoch {epoch}: val={vs:.4f} best={best:.4f} patience={counter}",flush=True)
torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),best=best,bstate=bstate,
                counter=counter,epoch=epoch),STATE)
with open(os.path.expanduser("~/mdn_scalers.pkl"),"wb") as f:
    pickle.dump(dict(sc_ac=sc_ac,sc_art=sc_art,held=held,keep=keep),f)
print("DONE" if counter>=10 else "MORE", f"epoch={epoch} best={best:.4f}")
