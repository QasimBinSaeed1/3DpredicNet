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
from unet import MobileNetV3, YOLO11mDepth
from utils.utils import plot_img_and_mask

import matplotlib.pyplot as plt
import cv2 

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

    return output.squeeze(0).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoint_epoch100_depth.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', default = "./predict/imgs2/", metavar='INPUT', help='Filenames of input images')
    # parser.add_argument('--output', '-o', default = "./predict/outdepths/", metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--input', '-i', default = "./data/yolo_data_440/val/rgb", metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default = "./resultsyolo/", metavar='OUTPUT', help='Filenames of output images')
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
    # convert to HWC
    mask = mask.transpose(1, 2, 0)

    # convert to RGB (BGR for opencv)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = glob.glob(args.input + "/*.png")
    out_files = [f.replace(args.input, args.output) for f in in_files]

    # net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    # net = MobileNetV3(classes=3)
    net = YOLO11mDepth(nc=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # state_dict = torch.load(args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values')
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    logging.info('Model loaded!')

    # create output dir if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Path could not be found, created {args.output}')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        # resize the image to the same size as the training images
        img = img.resize((400, 400), Image.BILINEAR)


        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            # result.save(out_filename)
            # lets save image using plt instead of PIL
            print("i:", result.shape)

            cv2.imwrite(out_filename, result*255)
            


        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)


    # real testing
    print("##"*30)
    print("real testing for Abs_rel, Sq_rel, RMSE, RMSE_log, Acc1, Acc2, Acc3")

    depth_true_path = "predict/gt_map"
    depth_pred_path = "predict/outdepths"

    depth_true = [cv2.imread(os.path.join(depth_true_path, f)) for f in os.listdir(depth_true_path)]
    depth_pred = [cv2.imread(os.path.join(depth_pred_path, f)) for f in os.listdir(depth_pred_path)]

    # resize depth_true to 400x400
    depth_true = [cv2.resize(d, (400, 400)) for d in depth_true]
    

    depth_true = np.array(depth_true)
    depth_pred = np.array(depth_pred)

    depth_true = depth_true.astype(np.float32)
    depth_pred = depth_pred.astype(np.float32)

    depth_true = depth_true / 255
    depth_pred = depth_pred / 255

    # add 1 to depth_true to avoid division by zero
    depth_true += 1
    depth_pred += 1

    print("depth_true shape:", depth_true.shape, "depth_pred shape:", depth_pred.shape)
    AbsRel = np.mean(np.abs(depth_pred-depth_true)/depth_true)
    RMSE = np.sqrt(np.mean(np.abs(depth_pred-depth_true)**2))
    RMSE_log = np.sqrt(np.mean(np.abs(np.log10(depth_pred)-np.log10(depth_true))**2))
    SqRel = np.mean(np.abs(depth_pred-depth_true)**2/depth_true)

    print(f' AbsRel: {AbsRel} \n RMSE: {RMSE} \n RMSE_log: {RMSE_log} \n SqRel: {SqRel} ')