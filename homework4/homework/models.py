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
            if in_channels != out_channels:
                self.skip = nn.Linear(in_channels, out_channels)
            else:
                self.skip = nn.Identity()
        
        def forward(self, x):
            return self.relu(self.norm(self.linear(x))) + self.skip(x)


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

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        '''
        self.register_buffer('input_mean', torch.as_tensor(INPUT_MEAN))
        self.register_buffer('input_std', torch.as_tensor(INPUT_STD))
        '''

        c = n_track*2*2
        '''
        layers = []
        for _ in range(3):
            layers.append(self.Block(c, 128))
            c = 128
        '''
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.n_track*2*2),
            self.Block(c, 256),
            self.Block(256, 256),
            self.Block(256, 128),
            #*layers,
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
        x = torch.cat([track_left, track_right], dim=1)
        #z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        return self.network(x).view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

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
        raise NotImplementedError


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
            return self.relu(y + self.bn(self.skip(x)))

    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


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
