import os, sys, csv, json
import numpy as np, torch
PROJ = os.path.expanduser("~/mnt/speech_variation_surprisal_flexibility")
sys.path.insert(0, PROJ)
from utils.corpora import CORPORA
from utils.models import replicate_after_length

BLOB = f"{PROJ}/.cache/features/phone/frmri__prop25.f16"
meta = json.load(open(BLOB + ".meta.json")); H, W = meta["frame_shape"]
mm = np.memmap(BLOB, dtype=np.float16, mode="r").reshape(-1, H, W)
idx = {}
with open(BLOB + ".index.csv") as fh:
    for r in csv.DictReader(fh):
        idx[(r["art_f"], round(float(r["start"]),3), round(float(r["end"]),3))] = (int(r["offset"]), int(r["length"]))

TARGET = ("/Volumes/Transcend/FR-MRI/P6/OTHER/S1/VIDEO_P6_S1.avi", 2.907, 2.937)
off, n = idx[TARGET]
frames = np.asarray(mm[off:off+n], dtype=np.float32)          # (T, H, W), already
                                                              # per-recording normalised
spec = CORPORA["frmri"]
model = spec.build_art_model(device="cpu")
obj = torch.load(f"{PROJ}/phone_based/encoder_training/ckpts/frmri__prop25/art_model_ckpt.pth",
                 map_location="cpu")
model.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
model.eval()

x, lengths, _ = spec.collate()([(torch.from_numpy(frames).float(), 0.0)])
with torch.no_grad():
    # mirror VideoAutoencoder.encode step by step so the intermediates can be drawn
    lens = model._prepare(x, lengths)
    xp = model._pad_spatial(replicate_after_length(x, lens, time_dim=2))
    xt, theta = model.stn(xp, lens, None)
    h = model.encoder_frame_head(model.encoder_cnn(xt))        # (1, 32, T, 8, 8)
    B, C, T, gh, gw = h.shape
    emb = model.encoder_proj(h.permute(0, 2, 1, 3, 4).reshape(B, T, C*gh*gw))   # (1, T, 256)
    z = model.trunk.encode(emb, lens)                          # (1, 64)
    recon = model.decode(z, T=x.shape[2], lengths=lens, theta=theta)

out = dict(
    original = x[0, 0].numpy(),                     # (T, H, W)
    transformed = xt[0, 0, :, :H, :W].numpy(),
    recon = recon[0, 0].numpy(),
    framehead = h[0].permute(1, 0, 2, 3).numpy(),   # (T, 32, 8, 8)
    embedding = emb[0].numpy(),                     # (T, 256)
    latent = z[0].numpy(),                          # (64,)
    theta = theta[0].numpy(),
    T = int(lens[0]),
)
np.savez(os.path.expanduser("~/fig2_cd.npz"), **out)
print("T =", out["T"], " emb", out["embedding"].shape, " latent", out["latent"].shape)
print("theta:", np.round(out["theta"], 4).tolist())
print("recon vs original  MSE: %.4f   corr: %.3f" % (
    float(((out["recon"]-out["original"])**2).mean()),
    float(np.corrcoef(out["recon"].ravel(), out["original"].ravel())[0, 1])))
