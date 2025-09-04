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
from unet import NET
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt

output_depths = np.zeros((180, 3, 400, 400))

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                index=0
                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        output_depths[index] = output.squeeze(0).numpy()

    return output.squeeze(0).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch152.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default = "./data/pothole600/test/rgb/", metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default = "./predict/outdepths/", metavar='OUTPUT', help='Filenames of output images')
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


def mask_to_image(mask: np.ndarray):
    # if isinstance(mask_values[0], list):
    #     out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    # elif mask_values == [0, 1]:
    #     out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    # else:
    #     out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # if mask.ndim == 3:
    #     mask = np.argmax(mask, axis=0)

    # for i, v in enumerate(mask_values):
    #     out[mask == i] = v
    # convert to HWC
    mask = mask.transpose(1, 2, 0)
    # return Image.fromarray((mask*255).astype(np.uint8))
    return mask


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = glob.glob(args.input + "/*.png")
    out_files = [f.replace(args.input, args.output) for f in in_files]

    # net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    net = NET(classes=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # state_dict = torch.load(args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values')
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    # create output dir if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            device=device,
                            index=i
                           )

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            # result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            # plot_img_and_mask(img, mask)
            figure = plt.figure(figsize=(10, 10))
            plt.axis('off')

            plt.imshow(result)

            # save the figure
            plt.savefig(out_files[i])


