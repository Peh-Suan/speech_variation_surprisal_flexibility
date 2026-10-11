"""
Variable-length autoencoders for articulatory and acoustic speech trajectories.

Five modalities, one latent contract:

    MFCC                  (B, T, F)          -> SequenceAutoencoder
    EMA                   (B, T, P, C)       -> SequenceAutoencoder
    annotated MRI coords  (B, T, P, C)       -> SequenceAutoencoder
    raw dynamic MRI       (B, 1, T, H, W)    -> VideoAutoencoder
    ultrasound + lips     (B, 2, T, H, W)    -> VideoAutoencoder(stn_mode="separate")

Both models are:  feature encoder -> GRU over time -> masked pooling -> MLP -> z
                  z -> phase-conditioned GRU -> feature decoder -> reconstruction

--------------------------------------------------------------------------------
WHAT CHANGED RELATIVE TO THE FIRST DRAFT
--------------------------------------------------------------------------------

FIX 1 - z is now a function of the segment alone, not of the batch it landed in.
        Two separate leaks were closed:

        (a) GroupNorm normalised over (C, D, H, W), so replicated pad frames
            entered the statistics. All normalisation is now applied per time
            frame (`FrameNorm`), so statistics never cross the time axis.

        (b) Conv3d zero-pads at the tensor's time edge, so the LONGEST sample in
            each batch saw zeros where shorter samples saw replicated frames.
            Every convolutional stack now runs inside `TimeMargin`, which
            replicate-pads both ends by the stack's temporal radius and trims
            afterwards. Boundary handling no longer depends on D_max.

        `tests.py::test_batch_invariance` asserts bitwise equality.

FIX 2 - the decoder's time axis is normalised per sample, not by the batch's
        D_max. `decode(z, D, lengths=...)` gives every sample t = 0..1 across its
        own valid frames. Padded positions are overwritten with a copy of the
        last valid frame, so decoded valid frames are batch-invariant too.

FIX 3 - capacity moved to where it can see the data. Channels now widen as
        resolution drops (16, 32, 64, 128). Each frame is reduced by a 1x1 conv
        to `frame_head_channels` on a coarse `frame_grid` (8x8 by default) and
        projected to `frame_embed_dim` before the GRU, so the flatten stays cheap
        while "where" survives. latent_dim defaults to 64, the head MLP to one
        512 layer. Note the trap this avoids: flattening 128 channels at 16x16
        makes the projection alone an 8.4M-parameter Linear.

Also: rigid/similarity STN parameterisation (an affine transform can rescale
articulation, which is signal, not nuisance), optional per-session instead of
per-token theta, aspect-corrected rotation, closed-form AMP-safe affine inverse,
masked attention pooling over the GRU outputs, and an optional duration head so
"is duration in z?" becomes an explicit decision rather than an accident.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "VideoAutoencoder",
    "SequenceAutoencoder",
    "masked_mse",
    "reconstruction_loss",
    "temporal_diff",
    "masked_variance",
    "affine_identity_loss",
    "duration_loss",
    "invert_affine_2d",
]


# ==============================================================================
# Length / padding helpers
# ==============================================================================

def _length_view(lengths, ndim):
    """(B,) -> shape broadcastable against a tensor with `ndim` dims."""
    return lengths.view([-1] + [1] * (ndim - 1))


def _time_view(n, time_dim, ndim, device):
    shape = [1] * ndim
    shape[time_dim] = n
    return torch.arange(n, device=device).view(shape)


def valid_mask(x, lengths, time_dim):
    """Bool mask, broadcastable against x, true on valid time positions."""
    n = x.shape[time_dim]
    return _time_view(n, time_dim, x.ndim, x.device) < _length_view(lengths, x.ndim)


def last_valid(x, lengths, time_dim):
    """The final valid slice of each sample, keeping a singleton time axis."""
    idx_shape = list(x.shape)
    idx_shape[time_dim] = 1
    idx = (lengths - 1).view([-1] + [1] * (x.ndim - 1)).expand(idx_shape)
    return x.gather(time_dim, idx)


def replicate_after_length(x, lengths, time_dim):
    """
    Overwrite padded time positions with a copy of the last valid one.

    Cheaper than the draft's full-tensor gather: the index tensor here has a
    singleton time axis (a few hundred KB instead of ~105 MB at B=8, D=100,
    128x128), and `torch.where` broadcasts it.
    """
    return torch.where(valid_mask(x, lengths, time_dim), x, last_valid(x, lengths, time_dim))


class TimeMargin(nn.Module):
    """
    Run a length-preserving conv stack with `margin` replicated frames glued to
    both ends of the time axis, then trim them off.

    This is FIX 1(b). Without it, the boundary a convolution sees at the end of a
    sequence depends on how much padding happens to follow it in this batch,
    which makes the encoder's output depend on batch composition.
    """

    def __init__(self, body, margin, time_dim=2):
        super().__init__()
        self.body = body
        self.margin = int(margin)
        self.time_dim = time_dim

    def forward(self, x):
        if self.margin == 0:
            return self.body(x)
        d = self.time_dim
        n = x.shape[d]
        rep = [-1] * x.ndim
        rep[d] = self.margin
        head = x.narrow(d, 0, 1).expand(rep)
        tail = x.narrow(d, n - 1, 1).expand(rep)
        y = self.body(torch.cat([head, x, tail], dim=d))
        return y.narrow(d, self.margin, n)


# ==============================================================================
# Normalisation that does not cross the time axis  (FIX 1a)
# ==============================================================================

def _num_groups(num_channels, max_groups=8):
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class FrameNorm(nn.Module):
    """
    GroupNorm applied independently to every time frame.

    Input may be (B, C, T, H, W) or (B, C, T). Statistics are computed within a
    frame only, so padded frames cannot influence valid ones.
    """

    def __init__(self, num_channels, max_groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(_num_groups(num_channels, max_groups), num_channels)

    def forward(self, x):
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            return self.gn(y).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        if x.ndim == 3:
            B, C, T = x.shape
            y = x.permute(0, 2, 1).reshape(B * T, C, 1)
            return self.gn(y).reshape(B, T, C).permute(0, 2, 1)
        raise ValueError(f"FrameNorm expects a 3D or 5D tensor, got {x.ndim}D.")


def _per_frame_2d(x, fn):
    """Apply a 2D op to every time frame of a (B, C, T, H, W) tensor."""
    B, C, T, H, W = x.shape
    y = fn(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
    h, w = y.shape[-2:]
    return y.reshape(B, T, -1, h, w).permute(0, 2, 1, 3, 4)


class AvgPoolHW(nn.Module):
    """Average-pool H/W, leaving the time axis untouched."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        return _per_frame_2d(x, lambda y: F.avg_pool2d(y, self.kernel_size, self.stride))


_ADAPTIVE_WEIGHTS = {}


def _adaptive_weights(in_size, out_size, device, dtype):
    """
    The averaging matrix adaptive pooling applies along one axis.

    Bin i covers [floor(i*in/out), ceil((i+1)*in/out)) and is averaged, which
    is exactly what F.adaptive_avg_pool2d does -- adjacent bins overlap by a
    row when the sizes do not divide, which is the case this reproduces.
    Cached: the sizes are fixed per model.
    """
    key = (in_size, out_size, str(device), dtype)
    cached = _ADAPTIVE_WEIGHTS.get(key)
    if cached is not None:
        return cached
    w = torch.zeros(out_size, in_size, device=device, dtype=dtype)
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = -((-(i + 1) * in_size) // out_size)
        w[i, start:end] = 1.0 / (end - start)
    _ADAPTIVE_WEIGHTS[key] = w
    return w


def _adaptive_avg_pool2d(x, out_hw):
    """
    Adaptive average pooling that also runs on Apple's GPU.

    MPS implements adaptive pooling only when the input size divides the output
    size, and refuses otherwise -- "Non-divisible input sizes are not
    implemented on MPS device yet" (pytorch#96056). Every corpus here hits it:
    the CNN leaves 9x9, 11x11, 17x17 or 7x14 going into an 8x8 (or 7x8) grid.

    Pooling along an axis is a linear map, so the same result comes from
    multiplying by the averaging matrix, which MPS does support. Checked
    against F.adaptive_avg_pool2d on all four shapes: max |difference| 2.4e-07
    in float32 and 6.7e-16 in float64 -- two and three times machine epsilon --
    and the gradients agree exactly. Non-MPS devices keep the original call.
    """
    if x.device.type != "mps" or (x.shape[-2] % out_hw[0] == 0
                                  and x.shape[-1] % out_hw[1] == 0):
        return F.adaptive_avg_pool2d(x, out_hw)
    wh = _adaptive_weights(x.shape[-2], out_hw[0], x.device, x.dtype)
    ww = _adaptive_weights(x.shape[-1], out_hw[1], x.device, x.dtype)
    y = torch.einsum("oh,nchw->ncow", wh, x)
    return torch.einsum("pw,ncow->ncop", ww, y)


class AdaptivePoolHW(nn.Module):
    """Pool H/W to a fixed coarse grid, leaving the time axis untouched."""

    def __init__(self, grid):
        super().__init__()
        self.grid = tuple(grid)

    def forward(self, x):
        if x.shape[-2:] == self.grid:
            return x
        return _per_frame_2d(x, lambda y: _adaptive_avg_pool2d(y, self.grid))


class ResizeHW(nn.Module):
    """Nearest-resize H/W to a fixed size, leaving the time axis untouched."""

    def __init__(self, hw):
        super().__init__()
        self.hw = tuple(hw)

    def forward(self, x):
        if x.shape[-2:] == self.hw:
            return x
        return _per_frame_2d(x, lambda y: F.interpolate(y, size=self.hw, mode="nearest"))


# ==============================================================================
# Affine helpers
# ==============================================================================

def _grid_sample_border(frames, grid, align_corners=False):
    """
    Bilinear grid_sample with border padding, in a form Apple's GPU accepts.

    MPS rejects padding_mode="border" outright -- "Unsupported Border padding
    mode" -- and PYTORCH_ENABLE_MPS_FALLBACK does not catch it, because it is
    an explicit rejection rather than a missing kernel. Without this the whole
    model falls back to CPU, which on a Mac is the difference between minutes
    and hours per epoch.

    Clamping the sampling grid to the outermost pixel CENTRES and then padding
    with zeros computes the same thing. Border padding returns the edge pixel
    for any coordinate past the edge; a clamped grid never goes past the edge,
    so the zero padding is never reached and every in-range coordinate
    interpolates as before. With align_corners=False the last pixel centre sits
    at 1 - 1/W in normalised coordinates, which is where the clamp goes.

    Checked against the border implementation on the four input shapes this
    project uses, with roughly 40-50% of grid points deliberately landing
    outside the frame: max |difference| 3.1e-06 in float32 and 7.1e-15 in
    float64. Falling by eight orders of magnitude with the precision is what
    identifies it as the same computation rounded differently, not a different
    result. Non-MPS devices keep the original call, so anything trained on CPU
    is bit-for-bit what it was before.
    """
    if frames.device.type == "mps":
        H, W = frames.shape[-2:]
        gx = grid[..., 0].clamp(-1.0 + 1.0 / W, 1.0 - 1.0 / W)
        gy = grid[..., 1].clamp(-1.0 + 1.0 / H, 1.0 - 1.0 / H)
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(frames, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=align_corners)
    return F.grid_sample(frames, grid, mode="bilinear",
                         padding_mode="border", align_corners=align_corners)


def invert_affine_2d(theta, eps=1e-6):
    """
    Closed-form inverse of a batch of 2x3 affine matrices in affine_grid
    convention. Unlike torch.linalg.inv this has an fp16 kernel, so it survives
    autocast; it is also cheaper and the gradient is exact.
    """
    A = theta[..., :2]
    t = theta[..., 2:]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    # sign-preserving guard: bounded rigid/similarity theta always has det > 0,
    # but a free affine could in principle approach singularity.
    sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    det = torch.where(det.abs() < eps, sign * eps, det)
    inv_det = (1.0 / det)[..., None, None]
    A_inv = inv_det * torch.stack([
        torch.stack([A[..., 1, 1], -A[..., 0, 1]], dim=-1),
        torch.stack([-A[..., 1, 0], A[..., 0, 0]], dim=-1),
    ], dim=-2)
    t_inv = -(A_inv @ t)
    return torch.cat([A_inv, t_inv], dim=-1)


def affine_identity_loss(theta):
    """Optional regulariser pulling theta toward identity."""
    theta = theta.reshape(-1, 2, 3)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            device=theta.device, dtype=theta.dtype)
    return F.mse_loss(theta, identity.expand_as(theta))


# ==============================================================================
# Sequence-level spatial STN
# ==============================================================================

class SequenceSpatialSTN2D(nn.Module):
    """
    One 2D spatial transform per sequence, applied identically to every frame.
    Time is never transformed.

    transform:
        "rigid"      - rotation + translation (3 params). Default: it cannot
                       rescale or shear, so it cannot absorb articulatory
                       magnitude, which for a variability study is signal.
        "similarity" - rigid + isotropic scale (4 params).
        "affine"     - bounded residual around identity (6 params).

    conditioning:
        "token"   - theta predicted from the sequence itself (per speech token).
        "session" - theta is a free parameter per recording session. Probe
                    placement, anatomy and camera geometry are constant within a
                    session, not within a token; a per-token theta is free to
                    normalise away genuine between-token differences.

    Rotation is aspect-corrected, so it is a true rotation in pixel space even
    when H != W (affine_grid works in [-1, 1] on both axes, where a naive
    rotation matrix shears non-square images).
    """

    N_PARAMS = {"rigid": 3, "similarity": 4, "affine": 6}

    def __init__(self, input_channels, transform="rigid", conditioning="token",
                 num_sessions=None, max_rotation=0.26, max_translation=0.15,
                 max_log_scale=0.1, max_affine_delta=0.25):
        super().__init__()
        if transform not in self.N_PARAMS:
            raise ValueError(f"transform must be one of {sorted(self.N_PARAMS)}")
        if conditioning not in ("token", "session"):
            raise ValueError("conditioning must be 'token' or 'session'")
        if conditioning == "session" and not num_sessions:
            raise ValueError("conditioning='session' requires num_sessions")

        self.input_channels = input_channels
        self.transform = transform
        self.conditioning = conditioning
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.max_log_scale = max_log_scale
        self.max_affine_delta = max_affine_delta
        n = self.N_PARAMS[transform]

        if conditioning == "session":
            self.session_params = nn.Embedding(num_sessions, n)
            nn.init.zeros_(self.session_params.weight)
            self.localization = None
            self.theta_head = None
        else:
            self.session_params = None
            self.localization = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
                nn.GroupNorm(_num_groups(32), 32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.GroupNorm(_num_groups(64), 64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.theta_head = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, n),
            )
            nn.init.zeros_(self.theta_head[-1].weight)
            nn.init.zeros_(self.theta_head[-1].bias)

    # -- theta ----------------------------------------------------------------

    @staticmethod
    def _masked_temporal_mean(x, lengths):
        mask = valid_mask(x, lengths, time_dim=2).to(x.dtype)
        summed = (x * mask).sum(dim=2)
        denom = lengths.to(x.dtype).view(-1, 1, 1, 1).clamp_min(1)
        return summed / denom

    def _params_to_theta(self, p, aspect, dtype):
        """aspect = H / W, used to make rotation a true pixel-space rotation."""
        if self.transform == "affine":
            identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                    device=p.device, dtype=dtype)
            return identity + self.max_affine_delta * torch.tanh(p).view(-1, 2, 3)

        angle = self.max_rotation * torch.tanh(p[:, 0])
        tx = self.max_translation * torch.tanh(p[:, 1])
        ty = self.max_translation * torch.tanh(p[:, 2])
        scale = (torch.exp(self.max_log_scale * torch.tanh(p[:, 3]))
                 if self.transform == "similarity" else torch.ones_like(angle))
        c, s = torch.cos(angle) * scale, torch.sin(angle) * scale
        a = torch.as_tensor(aspect, device=p.device, dtype=dtype)
        row0 = torch.stack([c, -s * a, tx], dim=-1)
        row1 = torch.stack([s / a, c, ty], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def estimate_theta(self, x, lengths, session_ids=None):
        H, W = x.shape[-2:]
        if self.conditioning == "session":
            if session_ids is None:
                raise ValueError("conditioning='session' requires session_ids")
            p = self.session_params(session_ids.to(x.device))
        else:
            p = self.theta_head(self.localization(self._masked_temporal_mean(x, lengths)))
        return self._params_to_theta(p, aspect=H / W, dtype=x.dtype)

    # -- warping --------------------------------------------------------------

    @staticmethod
    def apply_theta(x, theta):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        theta_f = theta.unsqueeze(1).expand(B, T, 2, 3).reshape(B * T, 2, 3)
        grid = F.affine_grid(theta_f, list(frames.shape), align_corners=False)
        warped = _grid_sample_border(frames, grid)
        return warped.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(self, x, lengths, session_ids=None):
        theta = self.estimate_theta(x, lengths, session_ids)
        return self.apply_theta(x, theta), theta

    def inverse(self, x, theta):
        return self.apply_theta(x, invert_affine_2d(theta))


class SeparateViewSTN2D(nn.Module):
    """One independent sequence-level transform per view (e.g. ultrasound, lips)."""

    def __init__(self, num_views, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.view_stns = nn.ModuleList(
            [SequenceSpatialSTN2D(input_channels=1, **kwargs) for _ in range(num_views)])

    def forward(self, x, lengths, session_ids=None):
        if x.shape[1] != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {x.shape[1]}.")
        warped, thetas = [], []
        for v, stn in enumerate(self.view_stns):
            w, th = stn(x[:, v:v + 1], lengths, session_ids)
            warped.append(w)
            thetas.append(th)
        return torch.cat(warped, dim=1), torch.stack(thetas, dim=1)  # (B, V, 2, 3)

    def inverse(self, x, theta):
        return torch.cat([stn.inverse(x[:, v:v + 1], theta[:, v])
                          for v, stn in enumerate(self.view_stns)], dim=1)


# ==============================================================================
# Masked pooling over GRU outputs
# ==============================================================================

class MaskedPool(nn.Module):
    """
    "last"      - final hidden state only (the draft's behaviour).
    "attention" - concat[final hidden state, attention-pooled, masked mean].
                  Keeps the order information the GRU built while giving the
                  head a view of the whole trajectory, not just its endpoint.
    """

    def __init__(self, dim, mode="attention"):
        super().__init__()
        if mode not in ("last", "attention"):
            raise ValueError("pooling must be 'last' or 'attention'")
        self.mode = mode
        self.out_dim = dim if mode == "last" else 3 * dim
        if mode == "attention":
            hidden = max(dim // 2, 1)
            self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, outputs, h_last, lengths):
        if self.mode == "last":
            return h_last
        T = outputs.shape[1]
        mask = (torch.arange(T, device=outputs.device).view(1, T) <
                lengths.view(-1, 1)).unsqueeze(-1)
        logits = self.score(outputs).masked_fill(~mask, float("-inf"))
        attn = (outputs * torch.softmax(logits, dim=1)).sum(dim=1)
        mean = (outputs * mask).sum(dim=1) / lengths.to(outputs.dtype).view(-1, 1).clamp_min(1)
        return torch.cat([h_last, attn, mean], dim=-1)


# ==============================================================================
# Shared trunk: GRU encoder + phase-conditioned GRU decoder
# ==============================================================================

def _mlp(sizes, final_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TemporalTrunk(nn.Module):
    """
    (B, T, frame_embed_dim) -> z, and z -> (B, T, gru_hidden).

    The decoder is phase-conditioned: it receives t = 0..1 across each sample's
    OWN valid frames (FIX 2). Because z is supplied at every step and t is given
    explicitly, the decoder is not autoregressive over its own outputs, so there
    is no teacher forcing and no exposure bias.
    """

    def __init__(self, frame_embed_dim, latent_dim=64, gru_hidden=256, gru_layers=1,
                 bidirectional=True, mlp_hidden=(512,), pooling="attention",
                 time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.encoder_gru = nn.GRU(frame_embed_dim, gru_hidden, num_layers=gru_layers,
                                  batch_first=True, bidirectional=bidirectional)
        self.pool = MaskedPool(gru_hidden * self.n_directions, mode=pooling)
        self.fc_enc = _mlp([self.pool.out_dim] + list(mlp_hidden) + [latent_dim])

        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim), nn.ReLU())
        self.decoder_gru = nn.GRU(time_embed_dim + latent_dim, gru_hidden,
                                  num_layers=gru_layers, batch_first=True)
        self.fc_dec_init = nn.Linear(latent_dim, gru_layers * gru_hidden)

    def encode(self, frames, lengths):
        B = frames.shape[0]
        packed = pack_padded_sequence(frames, lengths.detach().cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, h_n = self.encoder_gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        h_n = h_n.view(self.gru_layers, self.n_directions, B, self.gru_hidden)[-1]
        h_last = h_n.permute(1, 0, 2).reshape(B, self.n_directions * self.gru_hidden)
        return self.fc_enc(self.pool(out, h_last, lengths))

    def decode(self, z, T, lengths):
        B = z.shape[0]
        idx = torch.arange(T, device=z.device, dtype=z.dtype).view(1, T)
        denom = (lengths.to(z.dtype) - 1).view(B, 1).clamp_min(1)
        t = (idx / denom).unsqueeze(-1)                      # (B, T, 1), >1 where padded
        inp = torch.cat([self.time_embedding(t),
                         z.unsqueeze(1).expand(B, T, self.latent_dim)], dim=-1)
        h0 = self.fc_dec_init(z).view(B, self.gru_layers, self.gru_hidden)
        h_seq, _ = self.decoder_gru(inp, h0.permute(1, 0, 2).contiguous())
        return h_seq


# ==============================================================================
# Video autoencoder
# ==============================================================================

class VideoAutoencoder(nn.Module):
    """
    Raw dynamic MRI (B, 1, T, H, W) or ultrasound + lips (B, 2, T, H, W).

    Encoder: sequence-level spatial STN -> Conv3d (H/W pooling only)
             -> per-frame projection -> GRU -> masked pooling -> MLP -> z
    Decoder: z -> phase-conditioned GRU -> per-frame spatial map
             -> Conv3d + spatial upsampling -> inverse STN
    """

    def __init__(self, input_hw, latent_dim=64, input_channels=1,
                 encoder_channels=(16, 32, 64, 128), stem_pool=1,
                 frame_head_channels=32, frame_grid=(8, 8),
                 frame_embed_dim=256, gru_hidden=256, gru_layers=1,
                 bidirectional_encoder=True, mlp_hidden=(512,), pooling="attention",
                 time_embed_dim=64, stn_mode="shared", stn_transform="rigid",
                 stn_conditioning="token", num_sessions=None,
                 predict_duration=True, output_activation="sigmoid"):
        super().__init__()
        self.input_hw = tuple(input_hw)
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.encoder_channels = tuple(encoder_channels)
        self.stem_pool = stem_pool
        self.stn_mode = stn_mode

        H, W = self.input_hw
        self.n_pools = len(encoder_channels) - 1
        self.spatial_multiple = stem_pool * 2 ** self.n_pools
        m = self.spatial_multiple
        self.padded_hw = (math.ceil(H / m) * m, math.ceil(W / m) * m)
        self.encoded_hw = (self.padded_hw[0] // m, self.padded_hw[1] // m)
        # Coarse spatial map kept per frame. Small enough that flattening it is
        # cheap, large enough that "where" survives (a 1x1 global pool would not).
        self.frame_grid = (min(frame_grid[0], self.encoded_hw[0]),
                           min(frame_grid[1], self.encoded_hw[1]))
        self.frame_head_channels = frame_head_channels
        self.frame_feature_dim = frame_head_channels * self.frame_grid[0] * self.frame_grid[1]

        # -- STN ---------------------------------------------------------------
        stn_kwargs = dict(transform=stn_transform, conditioning=stn_conditioning,
                          num_sessions=num_sessions)
        if stn_mode == "none":
            self.stn = None
        elif stn_mode == "shared":
            self.stn = SequenceSpatialSTN2D(input_channels=input_channels, **stn_kwargs)
        elif stn_mode == "separate":
            self.stn = SeparateViewSTN2D(num_views=input_channels, **stn_kwargs)
        else:
            raise ValueError("stn_mode must be 'none', 'shared' or 'separate'")

        # -- conv encoder: widen as resolution drops (FIX 3) --------------------
        layers, in_c, n_temporal = [], input_channels, 0
        if stem_pool > 1:
            layers.append(AvgPoolHW(stem_pool))
        for i, out_c in enumerate(encoder_channels):
            layers += [nn.Conv3d(in_c, out_c, 3, padding=1), FrameNorm(out_c), nn.ReLU()]
            n_temporal += 1
            if i < len(encoder_channels) - 1:
                layers.append(AvgPoolHW(2))
            in_c = out_c
        self.encoder_cnn = TimeMargin(nn.Sequential(*layers), margin=n_temporal)
        # 1x1x1 conv: reduces channels without touching time, so it needs no margin.
        self.encoder_frame_head = nn.Sequential(
            nn.Conv3d(encoder_channels[-1], frame_head_channels, 1),
            FrameNorm(frame_head_channels), nn.ReLU(),
            AdaptivePoolHW(self.frame_grid))
        self.encoder_proj = nn.Linear(self.frame_feature_dim, frame_embed_dim)

        # -- temporal trunk ----------------------------------------------------
        self.trunk = TemporalTrunk(frame_embed_dim, latent_dim=latent_dim,
                                   gru_hidden=gru_hidden, gru_layers=gru_layers,
                                   bidirectional=bidirectional_encoder,
                                   mlp_hidden=mlp_hidden, pooling=pooling,
                                   time_embed_dim=time_embed_dim)
        self.duration_head = nn.Linear(latent_dim, 1) if predict_duration else None

        # -- conv decoder ------------------------------------------------------
        self.decoder_proj = nn.Sequential(
            nn.Linear(gru_hidden, frame_embed_dim), nn.ReLU(),
            nn.Linear(frame_embed_dim, self.frame_feature_dim))
        self.decoder_frame_head = nn.Sequential(
            ResizeHW(self.encoded_hw),
            nn.Conv3d(frame_head_channels, encoder_channels[-1], 1),
            FrameNorm(encoder_channels[-1]), nn.ReLU())
        rev = list(reversed(encoder_channels))
        dl, in_c, n_temporal = [], rev[0], 0
        for out_c in rev[1:]:
            dl += [nn.Conv3d(in_c, out_c, 3, padding=1), FrameNorm(out_c), nn.ReLU(),
                   nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")]
            n_temporal += 1
            in_c = out_c
        if stem_pool > 1:
            dl.append(nn.Upsample(scale_factor=(1, stem_pool, stem_pool), mode="nearest"))
        dl.append(nn.Conv3d(in_c, input_channels, 3, padding=1))
        n_temporal += 1
        if output_activation == "sigmoid":
            dl.append(nn.Sigmoid())
        elif output_activation == "tanh":
            dl.append(nn.Tanh())
        elif output_activation is not None:
            raise ValueError("output_activation must be 'sigmoid', 'tanh' or None")
        self.decoder_cnn = TimeMargin(nn.Sequential(*dl), margin=n_temporal)

    # -- input handling --------------------------------------------------------

    def _prepare(self, x, lengths):
        if x.ndim != 5:
            raise ValueError("Expected x with shape (B, C, T, H, W).")
        B, C, T, H, W = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels/views, got {C}.")
        if (H, W) != self.input_hw:
            raise ValueError(f"Expected H/W={self.input_hw}, got {(H, W)}.")
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=x.device)
        if lengths.shape != (B,):
            raise ValueError("lengths must have shape (B,).")
        if torch.any(lengths < 1) or torch.any(lengths > T):
            raise ValueError("lengths must lie in [1, T].")
        return lengths

    def _pad_spatial(self, x):
        H, W = x.shape[-2:]
        pad_h, pad_w = self.padded_hw[0] - H, self.padded_hw[1] - W
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

    # -- encode / decode -------------------------------------------------------

    def encode(self, x, lengths=None, session_ids=None, return_theta=False):
        lengths = self._prepare(x, lengths)
        x = replicate_after_length(x, lengths, time_dim=2)
        x = self._pad_spatial(x)

        theta = None
        if self.stn is not None:
            x, theta = self.stn(x, lengths, session_ids)

        h = self.encoder_frame_head(self.encoder_cnn(x))         # (B, Ch, T, gh, gw)
        B, C, T, h_, w_ = h.shape
        frames = h.permute(0, 2, 1, 3, 4).reshape(B, T, C * h_ * w_)
        z = self.trunk.encode(self.encoder_proj(frames), lengths)
        return (z, theta) if return_theta else z

    def decode(self, z, T, lengths=None, theta=None):
        B = z.shape[0]
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=z.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=z.device)

        h_seq = self.trunk.decode(z, T, lengths)                 # (B, T, gru_hidden)
        feats = self.decoder_proj(h_seq)
        gh, gw = self.frame_grid
        feats = feats.view(B, T, self.frame_head_channels, gh, gw).permute(0, 2, 1, 3, 4)
        # Freeze the padded tail so decoded VALID frames do not depend on D_max.
        feats = replicate_after_length(feats, lengths, time_dim=2)

        out = self.decoder_cnn(self.decoder_frame_head(feats))
        if theta is not None and self.stn is not None:
            out = self.stn.inverse(out, theta)
        H, W = self.input_hw
        return out[:, :, :, :H, :W]

    # -- convenience -----------------------------------------------------------

    def predicted_log_duration(self, z):
        if self.duration_head is None:
            raise RuntimeError("Model was built with predict_duration=False.")
        return self.duration_head(z).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, x, lengths=None, session_ids=None):
        was_training = self.training
        self.eval()
        try:
            return self.encode(x, lengths=lengths, session_ids=session_ids)
        finally:
            self.train(was_training)

    def forward(self, x, lengths=None, session_ids=None,
                return_latent=False, return_theta=False, return_duration=False):
        lengths = self._prepare(x, lengths)
        z, theta = self.encode(x, lengths=lengths, session_ids=session_ids, return_theta=True)
        recon = self.decode(z, T=x.shape[2], lengths=lengths, theta=theta)
        extras = []
        if return_latent:
            extras.append(z)
        if return_theta:
            extras.append(theta)
        if return_duration:
            extras.append(self.predicted_log_duration(z))
        return (recon, *extras) if extras else recon


# ==============================================================================
# Sequence autoencoder: MFCC, EMA, annotated MRI coordinates
# ==============================================================================

class SequenceAutoencoder(nn.Module):
    """
    MFCC (B, T, F) or coordinates (B, T, P, C).

    Structurally parallel to VideoAutoencoder and subject to the same two fixes:
    normalisation never crosses the time axis, and the conv stack sees replicated
    temporal boundaries regardless of the batch's padded length.
    """

    def __init__(self, feature_shape, latent_dim=64, conv_channels=(128, 128, 128),
                 frame_embed_dim=256, gru_hidden=256, gru_layers=1,
                 bidirectional_encoder=True, mlp_hidden=(512,), pooling="attention",
                 time_embed_dim=64, predict_duration=True):
        super().__init__()
        self.feature_shape = tuple(feature_shape) if hasattr(feature_shape, "__iter__") \
            else (int(feature_shape),)
        self.feature_dim = int(math.prod(self.feature_shape))
        self.latent_dim = latent_dim
        self.conv_channels = tuple(conv_channels)

        layers, in_c, n_temporal = [], self.feature_dim, 0
        for out_c in conv_channels:
            layers += [nn.Conv1d(in_c, out_c, 3, padding=1), FrameNorm(out_c), nn.ReLU()]
            n_temporal += 1
            in_c = out_c
        self.encoder_cnn = TimeMargin(nn.Sequential(*layers), margin=n_temporal, time_dim=2)
        self.encoder_proj = nn.Linear(conv_channels[-1], frame_embed_dim)

        self.trunk = TemporalTrunk(frame_embed_dim, latent_dim=latent_dim,
                                   gru_hidden=gru_hidden, gru_layers=gru_layers,
                                   bidirectional=bidirectional_encoder,
                                   mlp_hidden=mlp_hidden, pooling=pooling,
                                   time_embed_dim=time_embed_dim)
        self.duration_head = nn.Linear(latent_dim, 1) if predict_duration else None

        self.decoder_proj = nn.Sequential(
            nn.Linear(gru_hidden, frame_embed_dim), nn.ReLU(),
            nn.Linear(frame_embed_dim, conv_channels[-1]))
        rev = list(reversed(conv_channels))
        dl, in_c, n_temporal = [], rev[0], 0
        for out_c in rev[1:]:
            dl += [nn.Conv1d(in_c, out_c, 3, padding=1), FrameNorm(out_c), nn.ReLU()]
            n_temporal += 1
            in_c = out_c
        dl.append(nn.Conv1d(in_c, self.feature_dim, 3, padding=1))
        n_temporal += 1
        self.decoder_cnn = TimeMargin(nn.Sequential(*dl), margin=n_temporal, time_dim=2)

    def _prepare(self, x, lengths):
        if x.shape[2:] != self.feature_shape:
            raise ValueError(f"Expected feature shape {self.feature_shape}, got {tuple(x.shape[2:])}.")
        B, T = x.shape[:2]
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=x.device)
        if lengths.shape != (B,) or torch.any(lengths < 1) or torch.any(lengths > T):
            raise ValueError("lengths must have shape (B,) with values in [1, T].")
        return lengths

    def encode(self, x, lengths=None):
        lengths = self._prepare(x, lengths)
        B, T = x.shape[:2]
        h = x.reshape(B, T, self.feature_dim).transpose(1, 2)     # (B, F, T)
        h = replicate_after_length(h, lengths, time_dim=2)
        h = self.encoder_cnn(h).transpose(1, 2)                   # (B, T, C)
        return self.trunk.encode(self.encoder_proj(h), lengths)

    def decode(self, z, T, lengths=None):
        B = z.shape[0]
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=z.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=z.device)
        h = self.decoder_proj(self.trunk.decode(z, T, lengths)).transpose(1, 2)
        h = replicate_after_length(h, lengths, time_dim=2)
        out = self.decoder_cnn(h).transpose(1, 2)                 # (B, T, F)
        return out.reshape(B, T, *self.feature_shape)

    def predicted_log_duration(self, z):
        if self.duration_head is None:
            raise RuntimeError("Model was built with predict_duration=False.")
        return self.duration_head(z).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, x, lengths=None):
        was_training = self.training
        self.eval()
        try:
            return self.encode(x, lengths=lengths)
        finally:
            self.train(was_training)

    def forward(self, x, lengths=None, return_latent=False, return_duration=False):
        lengths = self._prepare(x, lengths)
        z = self.encode(x, lengths=lengths)
        recon = self.decode(z, T=x.shape[1], lengths=lengths)
        extras = []
        if return_latent:
            extras.append(z)
        if return_duration:
            extras.append(self.predicted_log_duration(z))
        return (recon, *extras) if extras else recon


# ==============================================================================
# Losses
# ==============================================================================

def masked_mse(prediction, target, lengths, time_dim):
    """
    Reconstruction MSE over valid time positions only.

    time_dim=2 for video (B, C, T, H, W) and for (B, F, T);
    time_dim=1 for sequence tensors shaped (B, T, ...).
    """
    lengths = torch.as_tensor(lengths, device=prediction.device, dtype=torch.long)
    mask = valid_mask(prediction, lengths, time_dim).to(prediction.dtype)
    per_frame = prediction[0].numel() / prediction.shape[time_dim]
    num = (((prediction - target) ** 2) * mask).sum()
    den = (mask.sum() * per_frame).clamp_min(1.0)
    return num / den


def temporal_diff(x, time_dim):
    """
    First difference along the time axis: (..., T, ...) -> (..., T-1, ...).

    `time_dim` is not optional on purpose. Video tensors are (B, C, T, H, W), so
    a hard-coded x[:, 1:] slices CHANNELS, not time: silently empty for
    single-channel MRI, and silently "lips minus ultrasound" for a 2-view tensor.
    """
    n = x.shape[time_dim]
    if n < 2:
        raise ValueError("temporal_diff needs at least 2 time steps.")
    return x.narrow(time_dim, 1, n - 1) - x.narrow(time_dim, 0, n - 1)


def masked_variance(x, lengths, time_dim):
    """Variance over valid time positions only, as a scalar."""
    mask = valid_mask(x, lengths, time_dim).to(x.dtype)
    n = mask.expand_as(x).sum().clamp_min(1.0)
    mean = (x * mask).sum() / n
    return (((x - mean) ** 2) * mask).sum() / n


def reconstruction_loss(prediction, target, lengths, time_dim,
                        alpha=0.5, normalize=True, scales=None):
    """
    Static + velocity reconstruction. Differencing is a high-pass filter in time,
    so the delta term reweights the objective toward the frame-to-frame movement
    that plain MSE lets the model average away. It is the loss-side analogue of
    delta cepstral features.

    Two details that matter in a padded batch:

    1. The delta of a padded sequence has a spurious step at the boundary
       (last valid frame -> zero pad), which in a smooth articulatory video is
       ~60x the size of real frame-to-frame motion and would dominate the term.
       Valid deltas number `lengths - 1`, and that is what is masked here.

    2. `alpha` on raw MSEs is an arbitrary knob, because the two terms measure
       fractions of very different variances: var(delta x) is several times
       smaller than var(x), while iid reconstruction error is AMPLIFIED ~2x by
       differencing. With normalize=True each term becomes a fraction of its own
       target's variance, so alpha is an interpretable trade-off between
       "static variance explained" and "velocity variance explained".

    `scales`: pass a fixed (static, delta) pair estimated once over the dataset
    to keep the normaliser from wobbling batch to batch. If None, the batch's own
    variances are used, which is fine but noisier.

    Returns (total, static, delta).
    """
    lengths = torch.as_tensor(lengths, device=prediction.device, dtype=torch.long)
    static = masked_mse(prediction, target, lengths, time_dim)

    if prediction.shape[time_dim] < 2:
        # a batch of single-frame tokens has no velocity to fit
        zero = static.new_zeros(())
        return alpha * static, static, zero

    d_lengths = (lengths - 1).clamp_min(0)
    d_pred = temporal_diff(prediction, time_dim)
    d_target = temporal_diff(target, time_dim)
    delta = masked_mse(d_pred, d_target, d_lengths, time_dim)

    if normalize:
        if scales is None:
            s_static = masked_variance(target, lengths, time_dim)
            s_delta = masked_variance(d_target, d_lengths, time_dim)
        else:
            s_static, s_delta = (torch.as_tensor(s, device=prediction.device,
                                                 dtype=prediction.dtype) for s in scales)
        static = static / s_static.clamp_min(1e-8)
        delta = delta / s_delta.clamp_min(1e-8)

    return alpha * static + (1 - alpha) * delta, static, delta


def duration_loss(pred_log_duration, durations_seconds):
    """
    Auxiliary head making the duration question explicit.

    Keep this term if z should encode the complete realisation including its
    duration; drop it (predict_duration=False) if z should be trajectory SHAPE
    and duration should travel as a separate scalar. Pass seconds, not frames:
    your modalities run at different frame rates, so a duration in frames is not
    comparable across them.
    """
    target = torch.as_tensor(durations_seconds, device=pred_log_duration.device,
                             dtype=pred_log_duration.dtype)
    return F.mse_loss(pred_log_duration, torch.log(target.clamp_min(1e-6)))

class MDN(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        n_components=5,
        hidden_dim=128,
        sigma_min=0.01,
        sigma_init=1.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_components = n_components
        self.sigma_min = sigma_min

        # x_dim == 0 -> unconditional. Feed a constant so the trunk still runs and
        # the heads still produce (pi, mu, sigma), but they cannot depend on x.
        in_dim = x_dim if x_dim > 0 else 1
        if x_dim == 0:
            self.register_buffer("const", torch.ones(1, 1))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components * y_dim)
        self.raw_sigma_head = nn.Linear(hidden_dim, n_components * y_dim)

        # Start at sigma ~= sigma_init rather than at softplus(0) = 0.69. On
        # standardised targets sigma_init=1.0 starts the mixture at the marginal
        # scale, which is a much better place to descend from.
        inv_softplus = math.log(math.expm1(max(sigma_init - sigma_min, 1e-6)))
        nn.init.zeros_(self.raw_sigma_head.weight)
        nn.init.constant_(self.raw_sigma_head.bias, inv_softplus)

    def forward(self, x):
        """x: (B, x_dim). When x_dim == 0, pass a (B, 0) tensor -- only its
        batch size is used."""

        if self.x_dim == 0:
            h = self.net(self.const.expand(x.shape[0], -1))
        else:
            h = self.net(x)

        log_pi = F.log_softmax(self.pi_head(h), dim=1)

        mu = self.mu_head(h).view(-1, self.n_components, self.y_dim)

        raw = self.raw_sigma_head(h).view(-1, self.n_components, self.y_dim)
        sigma = self.sigma_min + F.softplus(raw)  # > 0, gradient never dies

        return log_pi, mu, sigma

    def log_prob(self, x, y):
        """x: (B, x_dim), y: (B, y_dim) -> log q(y | x), shape (B,)"""

        log_pi, mu, sigma = self(x)

        y = y.unsqueeze(1)  # (B, 1, D), broadcasts over the K components
        z = (y - mu) / sigma

        log_comp = (
            -0.5 * (z**2).sum(dim=2)
            - torch.log(sigma).sum(dim=2)
            - 0.5 * self.y_dim * math.log(2.0 * math.pi)
        )

        return torch.logsumexp(log_pi + log_comp, dim=1)

    @torch.no_grad()
    def sample(self, x, n_samples=1):
        """-> (B, n_samples, y_dim). For the round-trip / millimetre diagnostics:
        draw alternative articulations compatible with an acoustic token, decode
        them, and measure the spread in sensor space."""

        log_pi, mu, sigma = self(x)
        B = mu.shape[0]

        k = torch.multinomial(log_pi.exp(), n_samples, replacement=True)  # (B, S)
        idx = k.unsqueeze(-1).expand(B, n_samples, self.y_dim)

        mu_s = torch.gather(mu, 1, idx)
        sigma_s = torch.gather(sigma, 1, idx)

        return mu_s + sigma_s * torch.randn_like(mu_s)

    @torch.no_grad()
    def floor_fraction(self, x, tol=0.01):
        """Fraction of fitted sigmas sitting within `tol` (relative) of sigma_min.

        Run this on a fitted model. If it is more than a few percent your floor is
        binding and it is compressing the between-phoneme differences you care
        about -- lower sigma_min and refit."""

        _, _, sigma = self(x)
        return (sigma <= self.sigma_min * (1.0 + tol)).float().mean().item()