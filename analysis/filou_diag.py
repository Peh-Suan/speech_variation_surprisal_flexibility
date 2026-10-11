import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJ=os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility")
OUT=f"{PROJ}/analysis/_scratch/fig2"
d=np.load(os.path.expanduser("~/filou_sal.npz"))
L,U,anat=d["L"],d["U"],d["anat_l"]; D=len(L)
diff=L-U
f=lambda A:A.reshape(D,-1)

# is the difference just a rescaled copy of the /l/ map?
c_dl=[float(np.corrcoef(diff[k].ravel(),L[k].ravel())[0,1]) for k in range(D)]
# best scalar a_k with U ~ a_k * L, and what is left over
a=np.array([float(f(U)[k]@f(L)[k]/(f(L)[k]@f(L)[k])) for k in range(D)])
R=U-a[:,None,None]*L
frac=np.linalg.norm(f(R),axis=1)/np.linalg.norm(f(U),axis=1)
print("corr(diff, /l/ map) : median %.3f  min %.3f"%(np.median(c_dl),min(c_dl)))
print("best scalar a (U ~ a*L): median %.2f  range %.2f-%.2f"%(np.median(a),a.min(),a.max()))
print("residual ||U - aL|| / ||U||: median %.3f  min %.3f  max %.3f"%
      (np.median(frac),frac.min(),frac.max()))
print("frame counts: /l/ 5, /u/ 11 -> 1/T ratio %.2f ; observed gain %.2f"%(5/11,np.median(a)))

# scale-free difference: unit-normalise each map first, so only spatial
# redistribution survives, not the overall gain
Ln=L/np.linalg.norm(f(L),axis=1)[:,None,None]
Un=U/np.linalg.norm(f(U),axis=1)[:,None,None]
nd=Ln-Un
plt.rcParams.update({"font.family":"sans-serif","font.size":9,
                     "savefig.bbox":"tight","savefig.pad_inches":0.03})
fig,axes=plt.subplots(4,5,figsize=(4.7,4.9))
for k,ax in enumerate(axes.ravel()):
    v=float(np.abs(nd[k]).max())
    ax.imshow(anat,cmap="gray",interpolation="bilinear")
    ax.imshow(nd[k],cmap="bwr",vmin=-v,vmax=v,alpha=0.62,interpolation="bilinear")
    ax.set_title(str(k+1),pad=2.0,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
fig.suptitle("Latent dimension",y=0.985,fontsize=10)
fig.subplots_adjust(wspace=0.06,hspace=0.14,top=0.925,bottom=0.01,left=0.01,right=0.99)
fig.savefig(f"{OUT}/fig2e_filou_diff_normalised.pdf")
fig.savefig(f"{OUT}/fig2e_filou_diff_normalised.png",dpi=600); plt.close(fig)
print("saved normalised difference")
