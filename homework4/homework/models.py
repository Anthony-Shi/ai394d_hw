from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.linear = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm(out_channels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            if in_channels != out_channels:
                self.skip = nn.Linear(in_channels, out_channels)
            else:
                self.skip = nn.Identity()
        
        def forward(self, x):
            y = self.relu(self.norm(self.linear(x)))
            return self.dropout(y) + self.skip(x)


    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.register_buffer("input_mean", torch.zeros(2))
        self.register_buffer("input_std", torch.zeros(2))
        self.register_buffer("output_mean", torch.zeros(2))
        self.register_buffer("output_std", torch.zeros(2))

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        c = n_track*2*2

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.n_track*2*2),
            self.Block(c, 256),
            self.Block(256, 128),
            nn.Linear(128, self.n_waypoints*2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track_left_norm = (track_left - self.input_mean) / self.input_std
        track_right_norm = (track_right - self.input_mean) / self.input_std
        x = torch.cat([track_left_norm, track_right_norm], dim=1)
        y = self.network(x).view(-1, self.n_waypoints, 2)
        return y * self.output_std + self.output_mean


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.in_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, q, x, attn_mask=None):
        x_norm = self.in_norm(x)
        q_norm = self.in_norm(q)
        x = q + self.attn(q_norm, x_norm, x_norm, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x


class PerceiverBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.encoder = TransformerLayer(embed_dim, num_heads)
        self.processor = TransformerLayer(embed_dim, num_heads)

    def forward(self, q, x):
        x = self.encoder(q, x)
        x = self.processor(x, x)
        return x
    

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.register_buffer("input_mean", torch.zeros(2))
        self.register_buffer("input_std", torch.zeros(2))
        self.register_buffer("output_mean", torch.zeros(2))
        self.register_buffer("output_std", torch.zeros(2))

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.latent = nn.Parameter(nn.init.trunc_normal_(
            torch.zeros(128, d_model),
            0,
            0.2,
            -1,
            1,
        ))
        self.embed = nn.Linear(2, d_model)
        self.latent_block = nn.ModuleList(
            [PerceiverBlock(d_model, 8) for _ in range(2)]
        )

        self.query = nn.Parameter(torch.rand(self.n_waypoints, d_model))
        self.decoder = TransformerLayer(d_model, 8)

        self.linear = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track_left_norm = (track_left - self.input_mean) / self.input_std
        track_right_norm = (track_right - self.input_mean) / self.input_std
        x = torch.cat([track_left_norm, track_right_norm], dim=1)

        x = self.embed(x)
        latent = self.latent.expand(x.shape[0], -1, -1)
        for latent_block in self.latent_block:
            x = latent_block(latent, x)

        query = self.query.expand(x.shape[0], -1, -1)
        x = self.decoder(query, x)
        x = self.linear(x).view(-1, self.n_waypoints, 2)
        return x * self.output_std + self.output_mean


class CNNPlanner(nn.Module):
    class DownSampleBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super().__init__()
            padding = (kernel_size - 1) // 2
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

        def forward(self, x):
            y = self.relu(self.bn(self.conv1(x)))
            y = self.relu(self.bn(self.conv2(y)))
            y = self.bn(self.conv3(y))
            return self.relu(y + self.bn(self.skip(x)))
        
    class UpSampleBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, output_padding=1):
            super().__init__()
            self.stride = stride
            padding = (kernel_size - 1) // 2
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, 1, padding)
            self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.relu(self.bn(self.conv1(x)))
            y = self.relu(self.bn(self.conv2(y)))
            y = self.bn(self.conv3(y))
            x = self.skip(nn.functional.interpolate(x, scale_factor=self.stride))
            return self.relu(y + self.bn(x))

    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.zeros(3))
        self.register_buffer("input_std", torch.zeros(3))
        self.register_buffer("output_mean", torch.zeros(2))
        self.register_buffer("output_std", torch.zeros(2))

        c_in, c_out = 3, 32
        conv_layers = []
        for _ in range(3):
            conv_layers.append(self.DownSampleBlock(c_in, c_out, stride=2))
            c_in = c_out
            c_out *= 2
        self.conv_net = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(c_out // 2, 256)
        self.linear2 = nn.Linear(256, n_waypoints*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        image_norm = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.pool(self.conv_net(image_norm))
        x = self.dropout(self.relu(self.linear1(x.squeeze())))
        x = self.linear2(x)
        y = x.view(-1, self.n_waypoints, 2)
        return y * self.output_std + self.output_mean


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
