"""
Minimal end-to-end usage: collate, train step, latent extraction.

The one piece the model cannot do for you is the collate_fn — padding is where
the length bookkeeping starts, and every invariance in model.py depends on
`lengths` being correct.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
 
from .models import (VideoAutoencoder, SequenceAutoencoder, masked_mse,
                   reconstruction_loss, duration_loss)
 
 
# ------------------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------------------
 
def collate_video(batch):
    """
    batch: list of (video, duration_seconds) with the video TIME-FIRST, matching
    every other extractor:
 
        (T, H, W)     single-channel, e.g. rtMRI
        (T, V, H, W)  multi-view, e.g. ultrasound + lips
 
    Returns x as (B, C, T, H, W), which is what VideoAutoencoder expects — the
    channel axis is inserted or moved here, so the extractors never have to know
    about it. Pads along time like collate_sequence does.
 
    Zero-padding is fine: the encoder overwrites the pad with the last valid
    frame internally, and the loss is masked.
    """
    vids, durs = zip(*batch)
    if vids[0].ndim not in (3, 4):
        raise ValueError(f"expected (T, H, W) or (T, V, H, W); got {tuple(vids[0].shape)}")
    # channel-first items have a constant axis 0 and a varying axis 1 — catch that
    # here rather than letting torch.stack fail with a shape message
    if len(vids) > 1 and len({v.shape[0] for v in vids}) == 1 \
            and len({v.shape[1] for v in vids}) > 1:
        raise ValueError(
            f"items look CHANNEL-first: {tuple(vids[0].shape)} has a constant axis 0 "
            f"and a varying axis 1. collate_video wants time first, (T, H, W) or "
            f"(T, V, H, W). If you stack views, use axis=1, not axis=0.")
 
    lengths = torch.tensor([v.shape[0] for v in vids], dtype=torch.long)
    T = int(lengths.max())
    padded = torch.stack([
        torch.cat([v, v.new_zeros(T - v.shape[0], *v.shape[1:])]) for v in vids])
 
    if padded.ndim == 4:            # (B, T, H, W)    -> (B, 1, T, H, W)
        padded = padded.unsqueeze(1)
    elif padded.ndim == 5:          # (B, T, V, H, W) -> (B, V, T, H, W)
        padded = padded.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(
            f"collate_video expects items shaped (T, H, W) or (T, V, H, W); "
            f"got {tuple(vids[0].shape)}. If your extractor returns channel-first "
            f"(C, T, H, W), move the channel axis or use axis=1 when stacking views.")
 
    return padded, lengths, torch.tensor(durs, dtype=torch.float32)
 
 
def collate_sequence(batch):
    """batch: list of (features (T_i, ...), duration_seconds)"""
    seqs, durs = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    T = int(lengths.max())
    padded = torch.stack([
        torch.cat([s, s.new_zeros(T - s.shape[0], *s.shape[1:])]) for s in seqs])
    return padded, lengths, torch.tensor(durs, dtype=torch.float32)
 
 
def collate_for(model):
    """
    The collate that matches a model, so the pairing cannot be got wrong.
 
    Both collates take TIME-FIRST items — (T, F), (T, P, C), (T, H, W) or
    (T, V, H, W) — which is what every extractor returns. The difference is only
    what the model wants at the other end: SequenceAutoencoder takes (B, T, ...)
    and VideoAutoencoder takes (B, C, T, H, W), so collate_video also restores
    the channel axis.
    """
    return collate_video if isinstance(model, VideoAutoencoder) else collate_sequence
 
 
# ------------------------------------------------------------------------------
# Train step
# ------------------------------------------------------------------------------
 
def train_step(model, opt, x, lengths, durations=None, time_dim=None,
               alpha=0.5, scales=None, duration_mode="probe", duration_weight=1.0,
               clip=5.0):
    """
    One optimisation step, for EITHER model. Replaces the earlier
    `video_train_step`, which was video-only and predated the velocity term.
 
    `time_dim` is inferred from the model class if you leave it None:
    2 for VideoAutoencoder (B, C, T, H, W), 1 for SequenceAutoencoder (B, T, ...).
 
    `alpha` / `scales` go to reconstruction_loss: static vs velocity, normalised
    by fixed dataset-level variances so alpha means the same thing throughout
    training. Get `scales` once from estimate_loss_scales().
 
    Returns a dict of the individual terms, so you can log them separately —
    watching static and delta apart is how you tell "reproducing a blurry
    average" from "actually tracking the movement".
 
    ------------------------------------------------------------------
    `duration_mode` is the explicit form of the duration question. The three
    states are genuinely different, and the difference is one `.detach()`:
 
      "off"     - the duration head is never in a backward pass. It stays at its
                  random initialisation and its output is meaningless. Use this
                  only if you have no use for the head at all.
 
      "probe"   - the head is trained on z.detach(), so gradients update the
                  head's own weights and STOP there. The representation is not
                  shaped by duration at all; the head's accuracy is a
                  MEASUREMENT of how much duration is linearly readable off z.
                  This is the default: it costs nothing and tells you something
                  you need to know before interpreting any latent distance.
 
      "shaping" - the head is trained on z itself, so the gradient reaches the
                  encoder and duration is pushed INTO z. Use this if z is meant
                  to be the complete realisation. Pass SECONDS, not frames: your
                  modalities run at different frame rates, so a duration encoded
                  in frames is not comparable across them.
 
    Neither "off" nor "probe" makes z duration-free — the encoder GRU always
    sees the true length. They leave the leak alone; they do not close it. To
    actually remove duration you need gradient reversal on this head, or
    time-normalised input, or to project the direction out afterwards. The
    recommended route is duration_mode="probe" plus latent_with_duration() at
    analysis time: duration in the representation at a weight you chose, and
    removable without retraining.
 
    Returns {"total", "static", "delta", "duration"}.
    """
    if duration_mode not in ("off", "probe", "shaping"):
        raise ValueError("duration_mode must be 'off', 'probe' or 'shaping'")
    if time_dim is None:
        time_dim = 2 if isinstance(model, VideoAutoencoder) else 1
 
    model.train()
    opt.zero_grad(set_to_none=True)
    recon, z = model(x, lengths, return_latent=True)[:2]
    total, static, delta = reconstruction_loss(recon, x, lengths, time_dim,
                                               alpha=alpha, scales=scales)
    dur = None
    if duration_mode != "off":
        if durations is None:
            raise ValueError(f"duration_mode={duration_mode!r} needs `durations`")
        # z.detach() is the whole mechanism: the head learns, the encoder does not.
        zz = z.detach() if duration_mode == "probe" else z
        dur = duration_loss(model.predicted_log_duration(zz), durations)
        # in probe mode the weight cannot reach the encoder, so it is left at 1
        total = total + (duration_weight if duration_mode == "shaping" else 1.0) * dur
 
    total.backward()
    # Clip the duration head SEPARATELY. clip_grad_norm_ over model.parameters()
    # pools every gradient into one norm, so in probe mode the head's gradient
    # would scale the encoder's update down even though z was detached — the
    # detach isolates the backward pass but not the clipping. Measured: 'off' and
    # 'probe' drifted apart by 2e-2 in encoder weights over 5 epochs.
    head = [p for n, p in model.named_parameters() if n.startswith("duration_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("duration_head")]
    torch.nn.utils.clip_grad_norm_(body, clip)
    if head:
        torch.nn.utils.clip_grad_norm_(head, clip)
    opt.step()
    return {"total": total.item(), "static": static.item(), "delta": delta.item(),
            "duration": None if dur is None else dur.item()}
 
 
def estimate_loss_scales(batches, time_dim=None, max_batches=50):
    """
    Dataset-level (var_static, var_delta) for reconstruction_loss(scales=...).
 
    Pass these instead of letting each batch normalise by its own variance: a
    per-batch normaliser wobbles, and short-token batches have a different
    velocity variance from long-token ones, so `alpha` would silently drift.
 
    `batches`: an iterable of (x, lengths), i.e. COLLATED tensors, not extractor
    output. Note the axis moves at the collate: extractor items are time-first
    (T, ...), but a collated video batch is (B, C, T, H, W) with time at axis 2,
    while a collated sequence batch is (B, T, ...) with time at axis 1.
 
    `time_dim` is inferred from the batch rank if omitted — 5-D means video, so
    2; anything else means sequence, so 1. Pass it explicitly if you ever feed a
    sequence model a 3-D per-frame feature, which would also make the batch 5-D.
 
    One pass over a few dozen batches is plenty; call it once and hard-code the
    numbers.
    """
    from .models import valid_mask, temporal_diff
    s = ss = n = 0.0
    ds = dss = dn = 0.0
    for i, (x, lengths) in enumerate(batches):
        if i >= max_batches:
            break
        if time_dim is None:
            time_dim = 2 if x.ndim == 5 else 1
        m = valid_mask(x, lengths, time_dim).expand_as(x).to(x.dtype)
        s += (x * m).sum().item(); ss += ((x ** 2) * m).sum().item(); n += m.sum().item()
        if x.shape[time_dim] >= 2:
            d = temporal_diff(x, time_dim)
            dm = valid_mask(d, (lengths - 1).clamp_min(0), time_dim).expand_as(d).to(d.dtype)
            ds += (d * dm).sum().item(); dss += ((d ** 2) * dm).sum().item()
            dn += dm.sum().item()
    var = ss / max(n, 1.0) - (s / max(n, 1.0)) ** 2
    dvar = dss / max(dn, 1.0) - (ds / max(dn, 1.0)) ** 2
    return var, dvar
 
 
def latent_with_duration(z, durations_seconds, weight=1.0, stats=None):
    """
    Bake duration into the representation WITHOUT a shaping loss and WITHOUT
    retraining: append one scaled log-duration coordinate to z.
 
    Why this rather than duration_mode="shaping":
      - the weight is explicit. Shaping sets a weight on the LOSS, not on the
        representation, so you cannot know what fraction of a latent distance is
        duration until you measure it afterwards.
      - it is one known coordinate, so duration-free distances are z_full[:, :-1]
        rather than a second training run.
      - seconds, so it means the same thing in the MFCC, EMA, MRI and ultrasound
        latent spaces despite their different frame rates.
 
    `weight` is in units of "an average latent dimension": weight=1 gives the
    duration coordinate the same standard deviation as the RMS per-dimension
    standard deviation of z. Contribution to squared Euclidean distance scales as
    weight**2, so weight=3 makes duration worth about nine average dimensions.
 
    Fit `stats` on your training split and reuse it everywhere, so distances stay
    on one scale across splits and modalities.
 
    Returns (z_full, stats).
    """
    d = torch.as_tensor(durations_seconds, dtype=z.dtype, device=z.device).view(-1, 1)
    logd = torch.log(d.clamp_min(1e-6))
    if stats is None:
        stats = (logd.mean(), logd.std().clamp_min(1e-6),
                 z.std(dim=0).pow(2).mean().sqrt().clamp_min(1e-6))
    mu, sd, z_scale = stats
    return torch.cat([z, weight * z_scale * (logd - mu) / sd], dim=1), stats
 
 
def strip_duration(z_full):
    """The duration-free view of the same representation. Both distances come
    from one trained model, so the ablation is a slice, not a retrain."""
    return z_full[:, :-1]
 
 
def duration_probe_r2(Z_train, dur_train, Z_test, dur_test, ridge=1e-2):
    """
    The number to actually report: how much of log-duration is linearly readable
    off z, measured on held-out tokens.
 
    The online probe in the train step is a lagging indicator, because z drifts
    under it while it fits. This is the clean version: freeze the encoder, take z
    for two disjoint splits, fit here, evaluate there. A probe fit and evaluated
    on the same tokens always looks better than it is.
 
    R^2 near 0 -> z carries little linearly-decodable duration.
    R^2 near 1 -> your latent distances are substantially a duration ruler.
 
    Note this measures LINEAR decodability. Duration could still be present
    nonlinearly; swap in a small MLP probe if you want a tighter upper bound.
    """
    y_tr = torch.log(dur_train.clamp_min(1e-6))
    y_te = torch.log(dur_test.clamp_min(1e-6))
    X_tr = torch.cat([Z_train, torch.ones(len(Z_train), 1)], dim=1)
    X_te = torch.cat([Z_test, torch.ones(len(Z_test), 1)], dim=1)
    A = X_tr.T @ X_tr + ridge * torch.eye(X_tr.shape[1])
    w = torch.linalg.solve(A, X_tr.T @ y_tr)
    resid = ((y_te - X_te @ w) ** 2).sum()
    total = ((y_te - y_te.mean()) ** 2).sum().clamp_min(1e-12)
    return (1 - resid / total).item()
 
 
@torch.no_grad()
def extract_latents(model, loader, device="cpu"):
    """Collect z (and durations) for a whole split. Safe to batch however you
    like — z no longer depends on batch composition (tests.py asserts this)."""
    model.eval()
    zs, durs = [], []
    for x, lengths, d in loader:
        zs.append(model.extract_features(x.to(device), lengths.to(device)).cpu())
        durs.append(d)
    return torch.cat(zs), torch.cat(durs)