import os, sys, csv, json, time
import numpy as np, torch
PROJ = os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility")
sys.path.insert(0, PROJ)
from utils.corpora import CORPORA

N_DIMS = 20          # the first 20 latent dimensions, as in the caption
BATCH  = 8
STATE  = os.path.expanduser("~/fig2_sal_state.npz")
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0   # seconds this call

BLOB = f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta = json.load(open(BLOB + ".meta.json")); H, W = meta["frame_shape"]
mm = np.memmap(BLOB, dtype=np.float16, mode="r").reshape(-1, H, W)
idx = {}
with open(BLOB + ".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"], round(float(r["start"]),3), round(float(r["end"]),3))] = (int(r["offset"]), int(r["length"]))

toks = []
with open(f"{PROJ}/phone_based/data_processing/dataset_construction/datasets/processed_frmri.csv") as fh:
    for r in csv.DictReader(fh):
        if "/P6/" not in r["art_f"]:
            continue
        for p in ("phone1", "phone2"):
            if r[p] == "l":
                k = (r["art_f"], round(float(r[f"{p}_start"]),3), round(float(r[f"{p}_end"]),3))
                if k in idx and k not in toks:
                    toks.append(k)
toks.sort()

spec = CORPORA["frmri"]
model = spec.build_art_model(device="cpu")
obj = torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",
                 map_location="cpu")
model.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if os.path.exists(STATE):
    st = np.load(STATE)
    acc, done, thetas = st["acc"], int(st["done"]), list(st["thetas"])
else:
    acc, done, thetas = np.zeros((N_DIMS, H, W), np.float64), 0, []

t0 = time.time()
while done < len(toks) and time.time() - t0 < BUDGET:
    batch = toks[done:done + BATCH]
    arrs = [torch.from_numpy(np.asarray(mm[idx[k][0]:idx[k][0]+idx[k][1]], np.float32))
            for k in batch]
    x, lengths, _ = spec.collate()([(a, 0.0) for a in arrs])
    x.requires_grad_(True)
    z, theta = model.encode(x, lengths, return_theta=True)
    thetas.extend(theta.detach().numpy().reshape(len(batch), -1).tolist())
    for k in range(N_DIMS):
        g, = torch.autograd.grad(z[:, k].sum(), x, retain_graph=(k < N_DIMS - 1))
        for b in range(len(batch)):
            mid = int(lengths[b]) // 2
            acc[k] += g[b, 0, mid].detach().numpy().astype(np.float64)
    done += len(batch)

np.savez(STATE, acc=acc, done=done, thetas=np.array(thetas, np.float32), total=len(toks))
print(f"{done}/{len(toks)} tokens  ({time.time()-t0:.0f}s this call)")
