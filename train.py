import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from data import Reconstruction3DDataLoader
from I3D_model import InceptionI3d
from loss import CosineLoss
from model import OcbeAndDecoder
from utils import (
    list_video_names,
    save_json,
    setup_seed,
    split_train_val_videos,
    write2txt,
)
from validation import is_better, validate_model


VALIDATION_EPOCHS = {"ped2": 50, "avenue": 50, "shanghai": 40}
FINAL_TRAINING_EPOCHS = {"ped2": 31, "avenue": 40, "shanghai": 16}


def dataset_paths(dataset_type):
    if dataset_type == "ped2":
        return os.path.join("UCSDped2", "Train"), ".tif"
    if dataset_type == "avenue":
        return os.path.join("Avenue", "Train"), ".jpg"
    return os.path.join("ShanghaiTech", "training", "frames"), ".jpg"


def load_teacher(path, device):
    encoder = InceptionI3d(num_classes=400, in_channels=3, dropout_keep_prob=0.5)
    pretrained = torch.load(path, map_location=device)
    state = encoder.state_dict()
    compatible = {k: v for k, v in pretrained.items() if k in state and v.shape == state[k].shape}
    state.update(compatible)
    encoder.load_state_dict(state)
    encoder.to(device).eval().requires_grad_(False)
    return encoder


def make_checkpoint(args, epoch, model, optimizer, train_videos, val_videos,
                    validation, best_record):
    return {
        "student_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": vars(args),
        "train_videos": list(train_videos),
        "val_videos": list(val_videos),
        "validation": validation,
        "best_validation": best_record,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_folder, extension = dataset_paths(args.dataset_type)
    all_videos = list_video_names(train_folder)
    if args.protocol == "validation":
        train_videos, val_videos = split_train_val_videos(
            all_videos, args.val_ratio, args.split_seed, args.dataset_type
        )
    else:
        train_videos, val_videos = all_videos, []

    run_name = "{}_{}_seed{}_lr{}_bs{}".format(
        args.dataset_type, args.protocol, args.seed, args.lr, args.batch_size
    )
    checkpoint_dir = os.path.join(args.checkpoint_path, run_name)
    log_dir = os.path.join(args.log_path, run_name)
    text_log_dir = os.path.join(args.text_log_path, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(text_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    save_json(os.path.join(checkpoint_dir, "config.json"), {
        "config": vars(args),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_labels_used": False,
        "selection_rule": (
            "AUC, then Sep, then normal objective"
            if args.protocol == "validation" else "predefined final epoch"
        ),
    })
    print("Video split: {} train / {} validation".format(
        len(train_videos), len(val_videos)
    ))

    encoder = load_teacher(args.teacher_path, device)
    model = OcbeAndDecoder(
        in_channels_list=[480, 832, 1024],
        mem_dim=args.mem_dim,
        shrink_thres=args.shrink_thres,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5
    )
    cosine_loss = CosineLoss()
    mse_loss = nn.MSELoss()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Reconstruction3DDataLoader(
        train_folder, transform, args.resize, args.resize,
        img_extension=extension, dataset=args.dataset_type, train=True,
        train_stride=args.train_stride, video_names=train_videos,
        pseudo_seed=args.seed, deterministic_pseudo=False,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True,
    )
    val_dataset = val_loader = None
    if args.protocol == "validation":
        val_dataset = Reconstruction3DDataLoader(
            train_folder, transform, args.resize, args.resize,
            img_extension=extension, dataset=args.dataset_type, train=True,
            train_stride=args.val_stride, video_names=val_videos,
            pseudo_seed=args.val_pseudo_seed, deterministic_pseudo=True,
        )
        val_loader = data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
        )

    start_epoch = args.start_epoch
    best_record = None
    if args.resume:
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint["student_state_dict"])
        optimizer.load_state_dict(
            checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_record = checkpoint.get("best_validation")

    start_time = time.time()
    for epoch in tqdm(range(start_epoch, args.epochs), ascii=True):
        encoder.eval()
        model.train()
        total_loss_value = 0.0
        batch_count = 0
        for images, pseudo_images in train_loader:
            images = images.to(device)
            pseudo_images = pseudo_images.to(device)
            with torch.no_grad():
                teacher_normal = encoder(images)
            student_normal = model(*teacher_normal)
            cosine_normal = sum(
                cosine_loss(teacher, student)
                for teacher, student in zip(teacher_normal, student_normal[:3])
            )
            mse_normal = sum(
                mse_loss(teacher, student)
                for teacher, student in zip(teacher_normal, student_normal[:3])
            )
            attention = student_normal[3]
            entropy = (-attention * torch.log(attention + 1e-12)).sum(dim=-1).mean()
            normal_loss = (
                cosine_normal + args.mse_weight * mse_normal
                + args.entropy_weight * entropy
            )

            with torch.no_grad():
                teacher_pseudo = encoder(pseudo_images)
            student_pseudo = model(*teacher_pseudo)
            pseudo_discrepancy = sum(
                cosine_loss(teacher, student)
                for teacher, student in zip(teacher_pseudo, student_pseudo[:3])
            )
            pseudo_loss = torch.clamp(args.margin - pseudo_discrepancy, min=0.0)
            loss = normal_loss + args.lambda_pseudo * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss_value += float(loss.item())
            batch_count += 1

        average_loss = total_loss_value / max(batch_count, 1)
        writer.add_scalar("train/loss", average_loss, epoch)
        validation = None
        improved = False
        if args.protocol == "validation" and (
            (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1
        ):
            validation = validate_model(
                encoder, model, val_loader, val_dataset, device, args.mse_weight
            )
            candidate = {
                "epoch": epoch,
                "aligned_scoring_auc": validation["aligned_scoring_auc"],
                "aligned_effect_size": validation["aligned_effect_size"],
                "normal_objective": validation["normal_objective"],
            }
            improved = is_better(candidate, best_record)
            if improved:
                best_record = candidate
            writer.add_scalar("validation/auc", validation["aligned_scoring_auc"], epoch)
            writer.add_scalar("validation/separation", validation["aligned_effect_size"], epoch)

        checkpoint = make_checkpoint(
            args, epoch, model, optimizer, train_videos, val_videos,
            validation, best_record,
        )
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pth"))
        if improved:
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
        write2txt(text_log_dir, {
            "epoch": epoch, "train_loss": average_loss,
            "validation": validation, "best": best_record,
        })

    if args.protocol == "full_train":
        final_checkpoint = make_checkpoint(
            args, args.epochs - 1, model, optimizer, train_videos, [], None, None
        )
        torch.save(final_checkpoint, os.path.join(checkpoint_dir, "final_model.pth"))

    summary = {
        "completed": True,
        "protocol": args.protocol,
        "epochs": args.epochs,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "best_validation": best_record,
        "test_labels_used": False,
        "training_hours": (time.time() - start_time) / 3600.0,
    }
    save_json(os.path.join(checkpoint_dir, "training_summary.json"), summary)
    writer.close()
    return run_name, best_record


def build_parser():
    parser = argparse.ArgumentParser(
        description="Leakage-free ST-RD-VAD training"
    )
    parser.add_argument("--dataset_type", choices=["ped2", "avenue", "shanghai"], default="ped2")
    parser.add_argument("--protocol", choices=["validation", "full_train"], default="validation")
    parser.add_argument("--teacher_path", default="I3D_rgb_imagenet.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_pseudo", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.0002)
    parser.add_argument("--mem_dim", type=int, default=100)
    parser.add_argument("--shrink_thres", type=float, default=0.0025)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--val_stride", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--val_pseudo_seed", type=int, default=2026)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--checkpoint_path", default="checkpoints")
    parser.add_argument("--log_path", default="logs")
    parser.add_argument("--text_log_path", default="logs_txt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_dir", default="")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.epochs is None:
        budgets = VALIDATION_EPOCHS if args.protocol == "validation" else FINAL_TRAINING_EPOCHS
        args.epochs = budgets[args.dataset_type]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    print(args)
    train(args)
