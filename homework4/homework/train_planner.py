import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    transform_pipeline: str = "default",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 2,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data = load_data("drive_data/val", transform_pipeline=transform_pipeline, shuffle=False)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metrics = {"train_err": [], "val_err": []}

    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()

        for sample in train_data:
            track_left = sample["track_left"]
            track_right = sample["track_right"]
            waypoints = sample["waypoints"]
            track_left, track_right, waypoints = track_left.to(device), track_right.to(device), waypoints.to(device)

            out = model(track_left, track_right)

            optim.zero_grad()
            #print(out.shape, waypoints.shape)
            loss = torch.nn.functional.mse_loss(out, waypoints)
            loss.backward()
            optim.step()

            with torch.no_grad():
                metrics["train_err"].append(loss.item())

            global_step += 1
        
        with torch.inference_mode():
            model.eval()

            for sample in val_data:
                track_left = sample["track_left"]
                track_right = sample["track_right"]
                waypoints = sample["waypoints"]
                track_left, track_right, waypoints = track_left.to(device), track_right.to(device), waypoints.to(device)

                pred = model(track_left, track_right)
                val_error = torch.abs(pred - waypoints).mean().item()
                metrics["val_err"].append(val_error)

                global_step += 1

        epoch_train_err = torch.as_tensor(metrics["train_err"]).mean()
        epoch_val_err = torch.as_tensor(metrics["val_err"]).mean()
        logger.add_scalar("train_error",
                            epoch_train_err,
                            epoch)
        logger.add_scalar("val_erruracy",
                            epoch_val_err,
                            epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_err={epoch_train_err:.4f} "
                f"val_err={epoch_val_err:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transform_pipeline", type=str, default="default")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=2)

    """
    Usage:
        python3 -m homework.train_planner --your_args here
    """

    # pass all arguments to train
    train(**vars(parser.parse_args()))