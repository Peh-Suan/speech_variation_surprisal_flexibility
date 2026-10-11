import os, sys, pickle
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility"); sys.path.insert(0,PROJ)
from utils.models import MDN
OUT=f"{PROJ}/analysis/_scratch/fig2"; os.makedirs(OUT,exist_ok=True)

d=np.load(f"{PROJ}/phone_based/flexibility_estimation/latents/aemri75anno__prop25.npz",allow_pickle=True)
S=pickle.load(open(os.path.expanduser("~/mdn_scalers.pkl"),"rb"))
keep=S["keep"]; sc_ac,sc_art,held=S["sc_ac"],S["sc_art"],S["held"]
art=d["art"][keep].astype(np.float32); ac=d["ac"][keep].astype(np.float32)
lab=d["labels"][keep]; sub=d["subjects"][keep]
st=torch.load(os.path.expanduser("~/mdn_state.pt"),map_location="cpu")
model=MDN(x_dim=64,y_dim=64,n_components=5,hidden_dim=256)
model.load_state_dict(st["bstate"]); model.eval()

# PCA on the raw articulatory latents, so the axes are in the latent space's
# own units as in the original figure
pca=PCA(n_components=2).fit(art)
W=pca.components_                      # (2, 64)
c=pca.mean_

acn=sc_ac.transform(ac).astype(np.float32)
with torch.no_grad():
    pi,mu,sg=model(torch.from_numpy(acn[held]))   # (log_pi, mu, sigma)
pi=pi.exp().numpy(); mu=mu.numpy(); sg=sg.numpy()   # forward returns LOG pi
scale,mean=sc_art.scale_,sc_art.mean_
mu_raw=mu*scale+mean; sg_raw=sg*scale               # back to latent units

M=np.einsum("ij,nkj->nki",W,mu_raw-c)               # (n,5,2) projected means
# pick the token whose mixture is most spread out in the PC plane
spread=np.array([max(np.linalg.norm(m[a]-m[b]) for a in range(5) for b in range(5))
                 for m in M])
# among the most clearly multimodal tokens, take the one whose actual
# production lands nearest a predicted component, as in the original figure
truths=np.einsum("ij,nj->ni",W,art[held]-c)
cand=np.argsort(-spread)[:25]
near=np.array([np.min(np.linalg.norm(M[i]-truths[i],axis=1)) for i in cand])
pick=int(cand[int(np.argmin(near))]); gi=held[pick]
print("token: %s /%s/ idx %d   spread %.0f   weights %s" %
      (sub[gi],lab[gi],gi,spread[pick],np.round(pi[pick],3).tolist()))

m2=M[pick]                                          # (5,2)
C2=np.einsum("ij,kj,lj->kil",W,sg_raw[pick]**2,W)   # (5,2,2) projected covariances
w=pi[pick]
truth=W@(art[gi]-c)

lo=np.minimum(m2.min(0),truth)-0.9*np.sqrt(np.array([C.diagonal() for C in C2]).max(0))
hi=np.maximum(m2.max(0),truth)+0.9*np.sqrt(np.array([C.diagonal() for C in C2]).max(0))
g1=np.linspace(lo[0],hi[0],160); g2=np.linspace(lo[1],hi[1],160)
G1,G2=np.meshgrid(g1,g2); P=np.stack([G1,G2],-1)
dens=np.zeros_like(G1)
for k in range(5):
    Ci=np.linalg.inv(C2[k]); diff=P-m2[k]
    q=np.einsum("...i,ij,...j->...",diff,Ci,diff)
    dens+=w[k]*np.exp(-0.5*q)/(2*np.pi*np.sqrt(np.linalg.det(C2[k])))

plt.rcParams.update({"font.family":"sans-serif","font.size":9})
fig=plt.figure(figsize=(7.2,6.2))
ax=fig.add_subplot(111,projection="3d")
ax.plot_surface(G1,G2,dens,cmap="viridis",linewidth=0,antialiased=True,
                rcount=160,ccount=160,alpha=0.95)
top=dens.max()*1.35
for k in range(5):
    ax.plot([m2[k,0]]*2,[m2[k,1]]*2,[0,top],ls="--",c="k",lw=0.7,zorder=5)
    ax.scatter(m2[k,0],m2[k,1],top,marker="X",s=90,c="magenta",
               edgecolors="none",zorder=6,
               label="Component mean" if k==0 else None)
ax.plot([truth[0]]*2,[truth[1]]*2,[0,top*1.12],ls="--",c="k",lw=0.7,zorder=5)
ax.scatter(truth[0],truth[1],top*1.12,marker="*",s=150,facecolors="none",
           edgecolors="k",linewidths=0.9,zorder=7,label="Actual production")
ax.set_xlabel("PC1",labelpad=8); ax.set_ylabel("PC2",labelpad=8)
ax.set_zlabel("Probability density",labelpad=8)
ax.set_zlim(0,top*1.25)
ax.view_init(elev=22,azim=-60)
ax.legend(loc="upper right",frameon=False,bbox_to_anchor=(1.02,0.98))
fig.savefig(f"{OUT}/fig3.pdf",bbox_inches="tight")
fig.savefig(f"{OUT}/fig3.png",dpi=600,bbox_inches="tight"); plt.close(fig)
print("peak density %.3e   PC1 range [%.0f, %.0f]   PC2 range [%.0f, %.0f]" %
      (dens.max(),lo[0],hi[0],lo[1],hi[1]))
print("explained variance PC1/PC2: %.3f %.3f" % tuple(pca.explained_variance_ratio_))
