import torch
import torch.nn as nn
import argparse
import os
import time
from data import *
from I3D_model import InceptionI3d
from torchvision import transforms
from model import OcbeAndDecoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss import *
from test import Test
from utils import setup_seed, write2txt


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_type == 'ped2':
        train_folder = os.path.join('UCSDped2', 'Train')
        test_folder = os.path.join('UCSDped2', 'Test')
    if args.dataset_type == 'avenue':
        train_folder = os.path.join('Avenue', 'Train')
        test_folder = os.path.join('Avenue', 'Test')
    if args.dataset_type == 'shanghai':
        train_folder = os.path.join('ShanghaiTech', 'training', 'frames')
        test_folder = os.path.join('ShanghaiTech', 'testing', 'frames')
    img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    run_time = "train_num" + str(args.num) + "_lr" + str(args.lr) + "_bs" + str(args.batch_size)

    writer = SummaryWriter(log_dir="./logs/I3D/" + run_time + "/")

    log_txt_name = "./logs_txt/" + 'train_num_' + str(args.num)
    if not os.path.exists(log_txt_name):
        os.makedirs(log_txt_name)

    # 加载教师网络
    encoder = InceptionI3d(num_classes=400, in_channels=3, dropout_keep_prob=0.5)
    pretrained_dict = torch.load('I3D_rgb_imagenet.pt', map_location=device)
    model_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)
    encoder.to(device)
    encoder.eval()

    ocbe_decoder = OcbeAndDecoder(
        in_channels_list=[480, 832, 1024],
        mem_dim=args.mem_dim,
        shrink_thres=args.shrink_thres,
    ).to(device)

    train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                               resize_height=args.resize, resize_width=args.resize,
                                               dataset=args.dataset_type,
                                               img_extension=img_extension, jump=[1],
                                               train=True, train_stride=args.train_stride)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers, drop_last=True)

    tic = time.time()

    cos_similarity = CosineLoss()
    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(ocbe_decoder.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)

    auroc_img_best = 0.0
    best_epoch = 0
    best_ckp_path = None

    if args.resume:
        checkpoint = torch.load(args.model_dir, map_location=device)
        ocbe_decoder.load_state_dict(checkpoint['student_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']

    for epoch in tqdm(range(args.start_epoch, args.epochs), ascii=True):
        encoder.eval()
        ocbe_decoder.train()
        train_loss_total = 0
        num_batches = 0

        for idx, (images, pseudo_images) in enumerate(train_batch):
            images = images.to(device)
            pseudo_images = pseudo_images.to(device)

            # ---- 正常路径 ----
            with torch.no_grad():
                t_feat1, t_feat2, t_feat3 = encoder(images)

            s_feat1, s_feat2, s_feat3, attention = ocbe_decoder(t_feat1, t_feat2, t_feat3)

            loss1 = cos_similarity(t_feat1, s_feat1)
            loss2 = cos_similarity(t_feat2, s_feat2)
            loss3 = cos_similarity(t_feat3, s_feat3)

            loss_mse1 = mse(t_feat1, s_feat1)
            loss_mse2 = mse(t_feat2, s_feat2)
            loss_mse3 = mse(t_feat3, s_feat3)

            entropy = -attention * torch.log(attention + 1e-12)
            entropy_loss = entropy.sum(dim=-1).mean()

            loss_normal = (loss1 + loss2 + loss3) + 0.1 * (loss_mse1 + loss_mse2 + loss_mse3) + 0.0002 * entropy_loss

            # ---- 伪异常路径 ----
            with torch.no_grad():
                t_feat1_p, t_feat2_p, t_feat3_p = encoder(pseudo_images)

            s_feat1_p, s_feat2_p, s_feat3_p, _ = ocbe_decoder(t_feat1_p, t_feat2_p, t_feat3_p)

            loss1_p = cos_similarity(t_feat1_p, s_feat1_p)
            loss2_p = cos_similarity(t_feat2_p, s_feat2_p)
            loss3_p = cos_similarity(t_feat3_p, s_feat3_p)
            loss_pseudo_sum = loss1_p + loss2_p + loss3_p

            loss_contrastive = torch.clamp(args.margin - loss_pseudo_sum, min=0.0)

            # ---- 联合损失 ----
            Total_Loss = loss_normal + args.lambda_pseudo * loss_contrastive

            optimizer.zero_grad()
            Total_Loss.backward()
            torch.nn.utils.clip_grad_norm_(ocbe_decoder.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_total += Total_Loss.item()
            num_batches += 1

        avg_loss = train_loss_total / num_batches if num_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("learning_rate", current_lr, epoch)
        writer.add_scalar("train_loss_total", train_loss_total, epoch)
        writer.add_scalar("avg_train_loss", avg_loss, epoch)

        if (args.test_interval > 0) and (epoch % args.test_interval == 0):
            ckp_path = os.path.join(args.checkpoint_path, "I3D" + run_time, "epoch" + str(epoch) + ".pth")
            parent_ckp_path = os.path.join(args.checkpoint_path, "I3D" + run_time)
            if not os.path.exists(parent_ckp_path):
                os.makedirs(parent_ckp_path)
            torch.save({'student_state_dict': ocbe_decoder.state_dict(),
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict()}, ckp_path)

            auroc_img = Test(ckp_dir=ckp_path, data_dir=test_folder, resize=args.resize,
                             dataset_type=args.dataset_type, k=0.1, smooth_sigma=9)

            writer.add_scalar("auroc_img", auroc_img, epoch)
            log_line = (
                f"{run_time} || epoch: {epoch} "
                f"|| total_loss: {train_loss_total:.4f} "
                f"|| avg_loss: {avg_loss:.4f} "
                f"|| lr: {current_lr:.6f} "
                f"|| auroc_img: {auroc_img:.4f}"
            )
            write2txt(log_txt_name, log_line)

            if auroc_img > auroc_img_best + 1e-4:
                best_ckp_path = ckp_path
                auroc_img_best = auroc_img
                best_epoch = epoch

            print(f"Validation AUROC: {auroc_img:.4f} (best: {auroc_img_best:.4f} at epoch {best_epoch})")

    toc = time.time()
    print('Total training time: {:.2f} h'.format((toc - tic) / 3600))
    print("Best AUROC: {:.4f} at epoch {}".format(auroc_img_best, best_epoch))

    best_ckp_path_final = os.path.join(args.checkpoint_path, "I3D" + run_time, "best_model.pth")
    try:
        if best_ckp_path is not None and os.path.exists(best_ckp_path):
            best_checkpoint = torch.load(best_ckp_path, map_location=device)
            torch.save(best_checkpoint, best_ckp_path_final)
        else:
            torch.save({'student_state_dict': ocbe_decoder.state_dict(),
                        'epoch': best_epoch,
                        'optimizer': optimizer.state_dict()}, best_ckp_path_final)
    except Exception as e:
        print(f"Save best_model.pth failed with error: {e}")
        torch.save({'student_state_dict': ocbe_decoder.state_dict(),
                    'epoch': best_epoch,
                    'optimizer': optimizer.state_dict()}, best_ckp_path_final)

    return run_time, auroc_img_best, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_dir', type=str, default='best_ckp_path')
    parser.add_argument('--dataset_type', type=str, default="ped2")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_pseudo', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--mem_dim', type=int, default=100)
    parser.add_argument('--shrink_thres', type=float, default=0.0025)
    parser.add_argument('--train_stride', type=int, default=1)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    model_name, auroc_img_best, best_epoch = train(args)
    print("Final Best AUROC: {:.4f} at epoch {}".format(auroc_img_best, best_epoch))