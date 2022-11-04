# python3 inference_cityscapes.py --arch deeplabv3+ --model savedModels/model_unet_40.pth --save True

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import segmentation_models_pytorch as smp
from dataset_cityscapes import cityscapesLoader
from segtransformer import segformer_mit_b3
from liteseg_model.liteseg import LiteSeg

import argparse
import config
import yaml
from addict import Dict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to the model")
ap.add_argument('-a', '--arch', default='unet', choices=['unet', 'manet', 'linknet', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+', 'manet','fpn', 'segformer-b3', 'liteseg', 'stdc1', 'stdc2', 'a2fpn'], help='Choose different semantic segmention architecture')
ap.add_argument("-s", '--save', default=False, type=bool, help='save predicted output')

args = vars(ap.parse_args())

# replace device accordingly,Prefer to do the
device = torch.device('cpu')

# replace with location of folder containing "gtFine" and "leftImg8bit"
path_data = config.CITYSCAPES_DATASET

n_classes = 19
batch_size = config.BATCH_SIZE
num_workers = config.NUM_WORKERS

val_data = cityscapesLoader(
    root = path_data,
    split='val'
    )

val_loader = DataLoader(
    val_data,
    batch_size = batch_size,
    num_workers = num_workers,
    #pin_memory = pin_memory  # gave no significant advantage
)

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 19
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

model = None;

if args["arch"] == "pan":
    # create segmentation model with pretrained encoder
    model = smp.PAN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "unet":
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "manet":
    model = smp.MAnet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "linknet":
    model = smp.Linknet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "pspnet":
    model = smp.PSPNet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "deeplabv3":
    model = smp.DeepLabV3(
        encoder_name=ENCODER,
        encoder_depth=5,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "manet":
    model = smp.MANet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "deeplabv3+":
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_depth=5,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "fpn":
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
elif args["arch"] == "segformer-b3":
    model = segformer_mit_b3(in_channels=3, num_classes=CLASSES)
elif args["arch"] == "liteseg":
    backbone_network = "mobilenet"

    CONFIG = Dict(yaml.load(open("liteseg_model/config/training.yaml"), Loader=yaml.Loader))
    model = LiteSeg.build(backbone_network, None, CONFIG, is_train=True, classes=CLASSES)

model_path = args["model"]
print("model_path: {}".format(model_path))
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

model.eval()
with torch.no_grad():
    for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):

        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        start = time.time()
        # model prediction
        val_pred = model(val_images)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for arch: {} on device: {} is {} ms ".format(args['arch'], device, elapsed_time))

        # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
        # considering predictions with highest scores for each pixel among 19 classes
        prediction = val_pred.data.max(1)[1].cpu().numpy()
        ground_truth = val_labels.data.cpu().numpy()

        # replace 100 to change number of images to print.
        # 500 % 100 = 5. So, we will get 5 predictions and ground truths
        if image_num % 10 == 0:
            # Model Prediction
            decoded_pred = val_data.decode_segmap(prediction[0])
            decode_gt = val_data.decode_segmap(ground_truth[0])
            #fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
            fig, ax = plt.subplots(1, 3, figsize=(20, 30))
            fig.set_dpi(100)

            original_image = val_images[0].data.cpu().numpy().astype("uint8")
            print("original_image.shape: {} ".format(original_image.shape))
            original_image = original_image.transpose(1, 2, 0)

            ax[0].imshow(original_image[:, :, ::-1])  # BGR to RGB
            ax[0].set_title('Original Image ')

            ax[1].imshow(decode_gt[:, :, ::-1])  # BGR to RGB
            ax[1].set_title('GroundTruth Image ')
            ax[2].imshow(decoded_pred[:, :, ::-1])  # BGR to RGB
            ax[2].set_title('Segmented Image')
            filename = "output_images/output_{}".format(image_num)
            if args['save']:
                plt.savefig(filename, dpi=100)
                plt.close(fig)
            else:
                plt.show()
