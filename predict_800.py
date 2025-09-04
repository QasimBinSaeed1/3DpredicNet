import time

import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet, MobileNetV3, SegFormer, YOLO11mDepth, YOLO11lDepth
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    
    global total_time
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        print("img.shape: ", img.shape)
        start_time = time.time()
        output = net(img)
        end_time = time.time()
        output = output.cpu()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        print("Elapsed time: ", elapsed_time, "fps: ", 1/elapsed_time)
        
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


#write a code to print FPS of the model on the test set





def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    # parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch200.pth', metavar='FILE',
    parser.add_argument('--model', '-m', default='./checkpoint_epoch100_depth.pth', metavar='FILE',
                            help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default = ".data/pothole600_real/test/rgb/", metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default = "./predict/output/", metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    total_time = 0
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = glob.glob(args.input + "/*")
    out_files = [f.replace(args.input, args.output) for f in in_files]

    # net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    net = YOLO11mDepth(in_channels=3, nc=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values')
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    batch = torch.rand((1, 3, 400, 400))
    if device.type == 'cuda':
        batch = batch.to(device=device, dtype=torch.float32)
        net.to(device=device)
        with torch.no_grad():
            net(batch)    

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        img = img.resize((400, 400), Image.BILINEAR)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            # result = result.resize((400, 400))

            # mask = mask.resize((800, 800), Image.BILINEAR)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

    logging.info('Done!')
    print("Total time: ", total_time)
    print("Number of images: ", len(in_files))
    FPS = len(in_files)/total_time
    print("FPS: ", FPS)


    