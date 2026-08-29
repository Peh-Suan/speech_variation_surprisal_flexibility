import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

"""
Encoders for MFCCs, EMA, and annotated MRI coordinates
"""
class SequenceAutoencoder(nn.Module):
    """
    Autoencoder for time-varying feature sequences.

    Supports:
        MFCC:
            x.shape = (B, T, F)
            feature_shape = F or (F,)

        EMA:
            x.shape = (B, T, P, C)
            feature_shape = (P, C)

        annotated MRI coordinates:
            x.shape = (B, T, P, C)
            feature_shape = (P, C)

    All non-time feature dimensions are flattened into channels for Conv1d,
    then restored after decoding.
    """

    def __init__(
        self,
        feature_shape,
        latent_dim,
        conv_channels=(512, 256, 128),
        conv_kernel_sizes=(5, 5, 3),
        mlp_hidden=(4096, 2048),
    ):
        super().__init__()

        if isinstance(feature_shape, int):
            feature_shape = (feature_shape,)

        self.feature_shape = tuple(feature_shape)
        self.input_dim = math.prod(self.feature_shape)
        self.latent_dim = latent_dim

        if len(conv_channels) != len(conv_kernel_sizes):
            raise ValueError(
                "conv_channels and conv_kernel_sizes must have the same length."
            )

        # ============================================================
        # Encoder Conv1d
        # ============================================================

        encoder_layers = []

        in_channels = self.input_dim

        for out_channels, kernel_size in zip(
            conv_channels,
            conv_kernel_sizes,
        ):
            encoder_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])

            in_channels = out_channels

        self.enc_conv = nn.Sequential(
            *encoder_layers
        )

        self.enc_pool = nn.AdaptiveAvgPool1d(1)

        bottleneck_channels = conv_channels[-1]

        # ============================================================
        # Encoder MLP
        #
        # 128 -> 4096 -> 2048 -> latent_dim
        # ============================================================

        enc_sizes = (
            [bottleneck_channels]
            + list(mlp_hidden)
            + [latent_dim]
        )

        self.fc_enc = self._make_mlp(
            enc_sizes,
            final_activation=False,
        )

        # ============================================================
        # Decoder MLP
        #
        # latent_dim -> 2048 -> 4096 -> 128
        # ============================================================

        dec_sizes = (
            [latent_dim]
            + list(reversed(mlp_hidden))
            + [bottleneck_channels]
        )

        self.fc_dec = self._make_mlp(
            dec_sizes,
            final_activation=True,
        )

        # ============================================================
        # Decoder Conv1d
        #
        # 128 -> 256 -> 512 -> input_dim
        # ============================================================

        decoder_channels = (
            list(reversed(conv_channels[:-1]))
            + [self.input_dim]
        )

        decoder_kernels = list(
            reversed(conv_kernel_sizes)
        )

        decoder_layers = []

        in_channels = bottleneck_channels

        for i, (out_channels, kernel_size) in enumerate(
            zip(decoder_channels, decoder_kernels)
        ):
            decoder_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

            # No BN/ReLU after the final reconstruction layer
            if i < len(decoder_channels) - 1:
                decoder_layers.extend([
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ])

            in_channels = out_channels

        self.dec_conv = nn.Sequential(
            *decoder_layers
        )

    @staticmethod
    def _make_mlp(
        sizes,
        final_activation=False,
    ):
        layers = []

        for i in range(len(sizes) - 1):

            layers.append(
                nn.Linear(
                    sizes[i],
                    sizes[i + 1],
                )
            )

            if (
                i < len(sizes) - 2
                or final_activation
            ):
                layers.append(
                    nn.ReLU()
                )

        return nn.Sequential(*layers)

    def _flatten_features(self, x):
        """
        (B, T, ...) -> (B, flattened_features, T)
        """

        B = x.shape[0]
        T = x.shape[1]

        observed_shape = tuple(x.shape[2:])

        if observed_shape != self.feature_shape:
            raise ValueError(
                f"Expected feature shape "
                f"{self.feature_shape}, "
                f"got {observed_shape}."
            )

        x = x.reshape(
            B,
            T,
            self.input_dim,
        )

        return x.transpose(1, 2)

    def encode(self, x):
        """
        Returns:
            z: (B, latent_dim)
        """

        x = self._flatten_features(x)

        h = self.enc_conv(x)

        h = self.enc_pool(
            h
        ).squeeze(-1)

        z = self.fc_enc(h)

        return z

    def decode(self, z, T):
        """
        z:
            (B, latent_dim)

        Returns reconstruction in the original feature shape.
        """

        B = z.shape[0]

        h = self.fc_dec(z)

        h = h.unsqueeze(-1).expand(
            -1,
            -1,
            T,
        )

        x_hat = self.dec_conv(h)

        x_hat = x_hat.transpose(1, 2)

        x_hat = x_hat.reshape(
            B,
            T,
            *self.feature_shape,
        )

        return x_hat

    def forward(self, x):

        T = x.shape[1]

        z = self.encode(x)

        x_hat = self.decode(
            z,
            T,
        )

        return x_hat, z

"""
Encoders for raw videos
"""
class AvgPoolHW(nn.Module):
    """
    Average-pool H/W while leaving D unchanged.

    Input:
        (B, C, D, H, W)
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):

        B, C, D, H, W = x.shape

        x = (
            x
            .permute(0, 2, 1, 3, 4)
            .reshape(B * D, C, H, W)
        )

        x = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )

        H2, W2 = x.shape[-2:]

        x = (
            x
            .reshape(B, D, C, H2, W2)
            .permute(0, 2, 1, 3, 4)
        )

        return x


class FlowWarp3D(nn.Module):
    """
    Learn a 3D deformation field for one view.

    Input:
        (B, C, D, H, W)

    Output:
        warped tensor with the same shape
    """

    def __init__(
        self,
        input_channels=1,
        flow_channels=3,
        cpu_fallback=False,
    ):
        super().__init__()

        self.cpu_fallback = cpu_fallback

        self.flow_predictor = nn.Sequential(
            nn.Conv3d(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv3d(
                32,
                flow_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        # Start from identity transformation.
        nn.init.zeros_(
            self.flow_predictor[-1].weight
        )
        nn.init.zeros_(
            self.flow_predictor[-1].bias
        )

    def forward(self, x):

        B, C, D, H, W = x.shape

        flow = self.flow_predictor(x)
        # (B, 3, D, H, W)

        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                -1, 1, D,
                device=x.device,
                dtype=x.dtype,
            ),
            torch.linspace(
                -1, 1, H,
                device=x.device,
                dtype=x.dtype,
            ),
            torch.linspace(
                -1, 1, W,
                device=x.device,
                dtype=x.dtype,
            ),
            indexing="ij",
        )

        base_grid = torch.stack(
            (grid_x, grid_y, grid_z),
            dim=-1,
        )
        # (D, H, W, 3)

        base_grid = (
            base_grid
            .unsqueeze(0)
            .expand(B, -1, -1, -1, -1)
        )
        # (B, D, H, W, 3)

        # Convert voxel displacement to normalized grid coordinates.
        norm_factor = torch.tensor(
            [
                W / 2,
                H / 2,
                D / 2,
            ],
            device=x.device,
            dtype=x.dtype,
        ).view(
            1, 3, 1, 1, 1
        )

        flow_norm = (
            flow / norm_factor
        ).permute(
            0, 2, 3, 4, 1
        )
        # (B, D, H, W, 3)

        sampling_grid = (
            base_grid + flow_norm
        )

        original_device = x.device

        if (
            self.cpu_fallback
            and x.device.type == "mps"
        ):
            warped = F.grid_sample(
                x.cpu(),
                sampling_grid.cpu(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            return warped.to(
                original_device
            )

        return F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

class SeparateViewFlowWarp3D(nn.Module):
    """
    Apply an independent FlowWarp3D to each input view.

    Input:
        (B, V, D, H, W)

    where V is the number of views.

    For BE-US:
        V = 2
        view 0 = ultrasound
        view 1 = lip video
    """

    def __init__(
        self,
        num_views,
        cpu_fallback=False,
    ):
        super().__init__()

        self.num_views = num_views

        self.view_warpers = nn.ModuleList([
            FlowWarp3D(
                input_channels=1,
                cpu_fallback=cpu_fallback,
            )
            for _ in range(num_views)
        ])

    def forward(self, x):

        B, V, D, H, W = x.shape

        if V != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} views, "
                f"got {V}."
            )

        warped_views = []

        for view_idx, warper in enumerate(
            self.view_warpers
        ):

            # Keep channel dimension:
            # (B, 1, D, H, W)
            view = x[
                :,
                view_idx:view_idx + 1,
            ]

            warped_view = warper(view)

            warped_views.append(
                warped_view
            )

        return torch.cat(
            warped_views,
            dim=1,
        )


class VideoAutoencoder(nn.Module):
    """
    Autoencoder for raw video / volumetric sequences.

    Input:
        (B, C, D, H, W)

    Examples:
        ultrasound + lip video
        dynamic MRI

    D is left unchanged by spatial pooling.
    """

    def __init__(
        self,
        input_shape,
        latent_dim=1024,
        input_channels=1,
        encoder_channels=(64, 32, 16, 4),
        mlp_hidden=(4096, 2048),
        use_mlp=True,
        stn_mode="shared",
        grid_sample_cpu_fallback=False,
    ):
        super().__init__()

        self.input_shape = tuple(input_shape)
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.use_mlp = use_mlp
        self.stn_mode = stn_mode

        D, H, W = self.input_shape

        # Number of H/W pooling operations:
        # for (64,32,16,4), there are 3.
        self.n_pools = len(encoder_channels) - 1

        self.spatial_multiple = 2 ** self.n_pools

        padded_H = (
            math.ceil(H / self.spatial_multiple)
            * self.spatial_multiple
        )

        padded_W = (
            math.ceil(W / self.spatial_multiple)
            * self.spatial_multiple
        )

        encoded_H = (
            padded_H // self.spatial_multiple
        )

        encoded_W = (
            padded_W // self.spatial_multiple
        )

        self.encoded_shape = (
            encoder_channels[-1],
            D,
            encoded_H,
            encoded_W,
        )

        encoded_numel = math.prod(
            self.encoded_shape
        )

        # ============================================================
        # Spatial transformers
        # ============================================================

        if self.stn_mode == "none":

            self.encoder_stn = nn.Identity()
            self.decoder_stn = nn.Identity()

        elif self.stn_mode == "shared":

            self.encoder_stn = FlowWarp3D(
                input_channels=input_channels,
                cpu_fallback=grid_sample_cpu_fallback,
            )

            self.decoder_stn = FlowWarp3D(
                input_channels=input_channels,
                cpu_fallback=grid_sample_cpu_fallback,
            )

        elif self.stn_mode == "separate":

            self.encoder_stn = SeparateViewFlowWarp3D(
                num_views=input_channels,
                cpu_fallback=grid_sample_cpu_fallback,
            )

            self.decoder_stn = SeparateViewFlowWarp3D(
                num_views=input_channels,
                cpu_fallback=grid_sample_cpu_fallback,
            )

        else:

            raise ValueError(
                f"Unknown stn_mode: {self.stn_mode!r}. "
                f"Expected 'none', 'shared', or 'separate'."
            )

        # ============================================================
        # Conv3D encoder
        # ============================================================

        encoder_layers = []

        in_channels = input_channels

        for i, out_channels in enumerate(
            encoder_channels
        ):

            encoder_layers.extend([
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm3d(
                    out_channels
                ),
                nn.ReLU(),
            ])

            # Pool after every block except the final one
            if i < len(encoder_channels) - 1:
                encoder_layers.append(
                    AvgPoolHW(
                        kernel_size=2,
                        stride=2,
                    )
                )

            in_channels = out_channels

        self.encoder = nn.Sequential(
            *encoder_layers
        )

        # ============================================================
        # MLP bottleneck
        # ============================================================

        if use_mlp:

            enc_sizes = (
                [encoded_numel]
                + list(mlp_hidden)
                + [latent_dim]
            )

            self.fc_enc = self._make_mlp(
                enc_sizes,
                final_activation=False,
            )

            dec_sizes = (
                [latent_dim]
                + list(reversed(mlp_hidden))
                + [encoded_numel]
            )

            self.fc_dec = self._make_mlp(
                dec_sizes,
                final_activation=False,
            )

        # ============================================================
        # Conv3D decoder
        # ============================================================

        decoder_layers = []

        reversed_channels = list(
            reversed(encoder_channels)
        )

        in_channels = reversed_channels[0]

        for out_channels in reversed_channels[1:]:

            decoder_layers.extend([
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm3d(
                    out_channels
                ),
                nn.ReLU(),
                nn.Upsample(
                    scale_factor=(1, 2, 2)
                ),
            ])

            in_channels = out_channels

        decoder_layers.extend([
            nn.ConvTranspose3d(
                in_channels,
                input_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        ])

        self.decoder = nn.Sequential(
            *decoder_layers
        )

    @staticmethod
    def _make_mlp(
        sizes,
        final_activation=False,
    ):

        layers = []

        for i in range(
            len(sizes) - 1
        ):

            layers.append(
                nn.Linear(
                    sizes[i],
                    sizes[i + 1],
                )
            )

            if (
                i < len(sizes) - 2
                or final_activation
            ):
                layers.append(
                    nn.ReLU()
                )

        return nn.Sequential(
            *layers
        )

    def _pad(self, x):

        _, _, D, H, W = x.shape

        mult = self.spatial_multiple

        pad_h = (
            mult - H % mult
        ) % mult

        pad_w = (
            mult - W % mult
        ) % mult

        if pad_h or pad_w:

            x = F.pad(
                x,
                (
                    0, pad_w,
                    0, pad_h,
                    0, 0,
                ),
                mode="replicate",
            )

        return x

    def _check_shape(self, x):

        observed = tuple(
            x.shape[2:]
        )

        if observed != self.input_shape:
            raise ValueError(
                f"Expected video shape "
                f"{self.input_shape}, "
                f"got {observed}."
            )

    def encode(
        self,
        x,
        as_vector=True,
    ):

        self._check_shape(x)

        x = self._pad(x)

        warped_x = self.encoder_stn(x)

        z_map = self.encoder(
            warped_x
        )

        if not self.use_mlp:

            if as_vector:
                return z_map.flatten(1)

            return z_map

        z = self.fc_enc(
            z_map.flatten(1)
        )

        if as_vector:
            return z

        return z_map

    def decode(self, z):

        B = z.shape[0]

        if self.use_mlp:

            z = self.fc_dec(z)

            z = z.view(
                B,
                *self.encoded_shape,
            )

        out = self.decoder(z)

        out = self.decoder_stn(out)

        D, H, W = self.input_shape

        return out[
            :,
            :,
            :D,
            :H,
            :W,
        ]

    def extract_features(
        self,
        x,
        as_vector=True,
        require_grad=False,
    ):

        was_training = self.training
        self.eval()

        context = (
            torch.enable_grad()
            if require_grad
            else torch.no_grad()
        )

        try:
            with context:
                return self.encode(
                    x,
                    as_vector=as_vector,
                )

        finally:
            self.train(
                was_training
            )

    def forward(
        self,
        x,
        return_warped_x=False,
        return_latent=False,
    ):

        self._check_shape(x)

        orig = x

        x = self._pad(x)

        warped_x = self.encoder_stn(x)

        z_map = self.encoder(
            warped_x
        )

        latent = None

        if self.use_mlp:

            latent = self.fc_enc(
                z_map.flatten(1)
            )

            z_dec = self.fc_dec(
                latent
            )

            z_dec = z_dec.view(
                x.shape[0],
                *self.encoded_shape,
            )

        else:
            z_dec = z_map

        out = self.decoder(
            z_dec
        )

        out = self.decoder_stn(
            out
        )

        D, H, W = self.input_shape

        out = out[
            :,
            :,
            :D,
            :H,
            :W,
        ]

        warped_x = warped_x[
            :,
            :,
            :D,
            :H,
            :W,
        ]

        diff = warped_x - orig

        if return_warped_x:

            if return_latent:
                return (
                    out,
                    diff,
                    latent,
                )

            return out, diff

        if return_latent:
            return out, latent

        return out

class MDN(nn.Module):

    def __init__(
        self,
        x_dim,
        y_dim,
        n_components=5,
        hidden_dim=128,
        sigma_min=0.05,
        sigma_max=10.0,
    ):

        super().__init__()

        self.n_components = n_components
        self.y_dim = y_dim

        self.logsigma_min = math.log(sigma_min)
        self.logsigma_max = math.log(sigma_max)

        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_head = nn.Linear(
            hidden_dim,
            n_components,
        )

        self.mu_head = nn.Linear(
            hidden_dim,
            n_components * y_dim,
        )

        self.logsigma_head = nn.Linear(
            hidden_dim,
            n_components * y_dim,
        )

    def forward(self, x):

        h = self.net(x)

        # ----------------------------------------
        # Mixture weights
        # ----------------------------------------

        log_pi = F.log_softmax(
            self.pi_head(h),
            dim=1,
        )

        # ----------------------------------------
        # Means
        # ----------------------------------------

        mu = self.mu_head(h).view(
            -1,
            self.n_components,
            self.y_dim,
        )

        # ----------------------------------------
        # Standard deviations
        # ----------------------------------------

        raw_logsigma = (
            self.logsigma_head(h)
            .view(
                -1,
                self.n_components,
                self.y_dim,
            )
        )

        # CRITICAL:
        # prevent variance collapse / overflow
        logsigma = torch.clamp(
            raw_logsigma,
            min=self.logsigma_min,
            max=self.logsigma_max,
        )

        return log_pi, mu, logsigma

    def log_prob(self, x, y):

        """
        x: (B, x_dim)
        y: (B, y_dim)

        Returns:
            log q_phi(y | x), shape (B,)
        """

        log_pi, mu, logsigma = self(x)

        # (B, D) -> (B, 1, D)
        # Broadcasting handles the K components;
        # no need to explicitly expand.
        y = y.unsqueeze(1)

        # ----------------------------------------
        # Gaussian log-density
        # ----------------------------------------

        inv_var = torch.exp(
            -2.0 * logsigma
        )

        sq_error = (
            (y - mu) ** 2
        )

        log_exp = -0.5 * (
            sq_error * inv_var
        ).sum(dim=2)

        log_det = logsigma.sum(
            dim=2
        )

        log_norm = (
            -0.5
            * self.y_dim
            * math.log(2.0 * math.pi)
            - log_det
        )

        log_comp = (
            log_norm + log_exp
        )

        # ----------------------------------------
        # Mixture probability in log space
        # ----------------------------------------

        return torch.logsumexp(
            log_pi + log_comp,
            dim=1,
        )