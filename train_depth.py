import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate_depth import evaluate
from unet import UNet, MobileNetV3, YOLO11lDepth
from utils.data_loading_depth import BasicDataset, CarvanaDataset
from utils.ssim import SSIMLoss

dir_img = Path('./data/pothole600_real/')
dir_mask = Path('./data/pothole600_real/')
dir_checkpoint = Path('./yolocheckpoints/')
torch.autograd.set_detect_anomaly(True)

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        dataset_name: str = 'pothole600'
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dataset_name, 'train', img_scale)
        val_dataset = CarvanaDataset(dataset_name, 'val', img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dataset_name, img_scale, mode='train')
        val_dataset = BasicDataset(dataset_name, img_scale, mode='val')

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    n_train = len(dataset)
    n_val = len(val_dataset)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='Ablation 440 Depth', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True, min_lr=1e-6, factor=0.95)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = SSIMLoss().to(device)
    global_step = 0
    current_val_loss = 0
    best_val_loss = 1e10

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        if epoch <= 100:
            optimizer.param_groups[0]['lr'] = 0.001
            learning_rate = 0.001
        elif epoch <= 150:
            optimizer.param_groups[0]['lr'] = 0.0005
            learning_rate= 0.0005
        else:
            optimizer.param_groups[0]['lr'] = 0.00005
            learning_rate = 0.00005
        epoch_loss = 0
        iou_train = 0
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, memory_format=torch.channels_last)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3  = evaluate(model, val_loader, device, amp)
                        scheduler.step(sq_rel)
                        current_val_loss = sum([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                        logging.info('Validation round:')
                        logging.info(f'abs_rel: {abs_rel:.3f} | sq_rel: {sq_rel:.3f} | rmse: {rmse:.3f} | rmse_log: {rmse_log:.3f} | a1: {a1:.3f} | a2: {a2:.3f} | a3: {a3:.3f}')
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'val abs_rel': abs_rel,
                                'val rmse': rmse,
                                'val rmse_log': rmse_log,
                                'val sq_rel': sq_rel,
                                # "batch": wandb.Image(torch.cat([true_masks, masks_pred, images], dim=0).cpu()),
                                'rgb': wandb.Image(images[0].cpu()),
                                # 'depth': wandb.Image(true_masks[0].cpu()),
                                # 'pred': wandb.Image(masks_pred[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].cpu()),
                                    'pred': wandb.Image(masks_pred[0].cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch
                                # **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            if current_val_loss < best_val_loss:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                # state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}_{:.4f}.pth'.format(epoch, current_val_loss)))
                logging.info(f'Checkpoint {epoch} saved!')
                best_val_loss = current_val_loss
            else:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                # state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}_depth.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    # parser.add_argument('--dataset', '-d', type=str, default='pothole600resizepng_new_tr_val400_400', help='Dataset directory')
    parser.add_argument('--dataset', '-d', type=str, default='yolo_data_440', help='Dataset directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # Initialize YOLO11mDepth model
    model = YOLO11lDepth(in_channels=3, nc=3)
    model = model.to(memory_format=torch.channels_last)
    
    # model = MobileNetV3(classes=3)
    # model = model.to(memory_format=torch.channels_last)

    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # try:
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        dataset_name=args.dataset
    )
    # except:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     # model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
