import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.classification_dataset import load_data as load_class_data
from .datasets.road_dataset import load_data as load_drive_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
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

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    if model_name == 'classifier':
        train_data = load_class_data("classification_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
        val_data = load_class_data("classification_data/val", transform_pipeline="aug", shuffle=False)
    else:
        train_data = load_drive_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
        val_data = load_drive_data("drive_data/val", transform_pipeline="aug", shuffle=False)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": [], "seg_train_acc": [], "depth_train_acc": [], "seg_val_acc": [], "depth_val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        if model_name == 'classifier':
            for img, label in train_data:
                img, label = img.to(device), label.to(device)

                out = model(img)

                optimizer.zero_grad()
                loss_val = torch.nn.functional.cross_entropy(out, label)
                loss_val.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_label = torch.argmax(out, dim=1)
                    train_accuracy = (pred_label == label).sum().item()
                    metrics["train_acc"].append(train_accuracy / batch_size)

                global_step += 1
        else:
            for sample in train_data:
                img = sample["image"]
                depth = sample["depth"]
                track = sample["track"]
                img, depth, track = img.to(device), depth.to(device), track.to(device)

                seg_out, depth_out = model(img)

                optimizer.zero_grad()
                seg_loss = torch.nn.functional.cross_entropy(seg_out, track)
                depth_loss = torch.nn.functional.mse_loss(depth_out, depth)
                total_loss = seg_loss + depth_loss
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_seg = torch.argmax(seg_out, dim=1)
                    seg_train_accuracy = (pred_seg == track).float().mean().item()
                    pred_depth = depth_out[0]
                    depth_train_accuracy = torch.abs(pred_depth - depth).mean().item()
                    metrics["seg_train_acc"].append(seg_train_accuracy)
                    metrics["depth_train_acc"].append(depth_train_accuracy)

                global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            if model_name == 'classifier':
                for img, label in val_data:
                    img, label = img.to(device), label.to(device)

                    out = model(img)
                    pred_label = torch.argmax(out, dim=1)
                    val_accuracy = (pred_label == label).sum().item()
                    metrics["val_acc"].append(val_accuracy / batch_size)

                    global_step += 1
            else:
                for sample in val_data:
                    img = sample["image"]
                    depth = sample["depth"]
                    track = sample["track"]
                    img, depth, track = img.to(device), depth.to(device), track.to(device)

                    seg_out, depth_out = model(img)
                    pred_seg = torch.argmax(seg_out, dim=1)
                    seg_val_accuracy = (pred_seg == track).float().mean().item()
                    pred_depth = depth_out[0]
                    depth_val_accuracy = torch.abs(pred_depth - depth).mean().item()
                    metrics["seg_val_acc"].append(seg_val_accuracy)
                    metrics["depth_val_acc"].append(depth_val_accuracy)

                    global_step += 1


        # log average train and val accuracy to tensorboard
        if model_name == 'classifier':
            epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
            epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
            logger.add_scalar("train_accuracy",
                                epoch_train_acc,
                                epoch)
            logger.add_scalar("val_accuracy",
                                epoch_val_acc,
                                epoch)

            # print on first, last, every 10th epoch
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                    f"train_acc={epoch_train_acc:.4f} "
                    f"val_acc={epoch_val_acc:.4f}"
                )
        else:
            epoch_seg_train_acc = torch.as_tensor(metrics["seg_train_acc"]).mean()
            epoch_depth_train_acc = torch.as_tensor(metrics["depth_train_acc"]).mean()
            epoch_seg_val_acc = torch.as_tensor(metrics["seg_val_acc"]).mean()
            epoch_depth_val_acc = torch.as_tensor(metrics["depth_val_acc"]).mean()
            logger.add_scalar("seg_train_accuracy",
                                epoch_seg_train_acc,
                                epoch)
            logger.add_scalar("depth_train_accuracy",
                                epoch_depth_train_acc,
                                epoch)
            logger.add_scalar("seg_val_accuracy",
                                epoch_seg_val_acc,
                                epoch)
            logger.add_scalar("depth_val_accuracy",
                                epoch_depth_val_acc,
                                epoch)

            # print on first, last, every 10th epoch
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                    f"seg_train_acc={epoch_seg_train_acc:.4f} "
                    f"depth_train_acc={epoch_depth_train_acc:.4f} "
                    f"seg_val_acc={epoch_seg_val_acc:.4f} "
                    f"depth_val_acc={epoch_depth_val_acc:.4f}"
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
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
