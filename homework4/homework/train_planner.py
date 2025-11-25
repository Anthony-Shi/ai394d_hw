import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

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

    # set up planner metric
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data = load_data("drive_data/val", transform_pipeline=transform_pipeline, shuffle=False)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0

    # normalize input data
    global_sum, global_sq_sum = torch.zeros(2, device=device), torch.zeros(2, device=device)
    n = 0
    for sample in train_data:
        image = sample["image"].to(device) # (b, c, h, w)
        track_left = sample["track_left"].to(device) # (b, 10, 2)
        track_right = sample["track_right"].to(device) # (b, 10, 2)
        waypoints = sample["waypoints"].to(device) # (b, 3, 2)

        points = torch.cat((track_left.reshape(-1, 2),
                    track_right.reshape(-1, 2),
                    waypoints.reshape(-1, 2)))
        global_sum += points.sum(dim=0)
        global_sq_sum += (points ** 2).sum(dim=0)
        n += points.shape[0]
    
    input_mean = global_sum / n
    input_std = ((global_sq_sum / n) - (input_mean ** 2)).sqrt()
    print(input_mean, input_std)

    for epoch in range(num_epoch):
        train_metric.reset()

        model.train()

        for sample in train_data:
            image = sample["image"].to(device) # (b, c, h, w)
            track_left = sample["track_left"].to(device) # (b, 10, 2)
            track_right = sample["track_right"].to(device) # (b, 10, 2)
            waypoints = sample["waypoints"].to(device) # (b, 3, 2)
            waypoints_mask = sample["waypoints_mask"].to(device) # (b, 3)

            track_left_norm, track_right_norm = (track_left - input_mean) / input_std, (track_right - input_mean) / input_std
            if model_name == "mlp_planner":
                out = model(track_left_norm, track_right_norm)
            elif model_name == "cnn_planner":
                out = model(image)

            optim.zero_grad()
            waypoints_norm = (waypoints - input_mean) / input_std
            loss = (out - waypoints_norm).abs()
            loss_masked = loss * waypoints_mask[..., None]
            n = waypoints_mask.sum()
            longitudinal_loss = loss_masked[..., 0].sum() / n
            lateral_loss = loss_masked[..., 1].sum() / n
            l1_loss = longitudinal_loss + lateral_loss
            l1_loss.backward()
            optim.step()
            
            train_metric.add(out, waypoints_norm, waypoints_mask)

            global_step += 1
        
        with torch.inference_mode():
            model.eval()

            for sample in val_data:
                track_left = sample["track_left"].to(device)
                track_right = sample["track_right"].to(device)
                waypoints = sample["waypoints"].to(device)
                waypoints_mask = sample["waypoints_mask"].to(device)
                
                track_left_norm, track_right_norm = (track_left - input_mean) / input_std, (track_right - input_mean) / input_std

                pred_norm = model(track_left_norm, track_right_norm)
                pred = (pred_norm * input_std) + input_mean
                val_metric.add(pred, waypoints, waypoints_mask)

                global_step += 1

        train_dict = train_metric.compute()
        val_dict = val_metric.compute()
        logger.add_scalar("train_longitudinal_error",
                            train_dict["longitudinal_error"],
                            epoch)
        logger.add_scalar("train_lateral_error",
                            train_dict["lateral_error"],
                            epoch)
        logger.add_scalar("val_longitudinal_error",
                            val_dict["longitudinal_error"],
                            epoch)
        logger.add_scalar("val_lateral_error",
                            val_dict["lateral_error"],
                            epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_err={train_dict["l1_error"]:.4f} "
                f"val_err={val_dict["l1_error"]:.4f}"
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