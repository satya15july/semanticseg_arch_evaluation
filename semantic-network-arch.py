# Usage: python3 semantic-network-arch.py --arch a2fpn --width 256 --height 256 --profiler all
import segmentation_models_pytorch as smp
from segtransformer import segformer_mit_b3
from liteseg_model.liteseg import LiteSeg
from bisenet.models.model_stages import BiSeNet
from A2FPN import A2FPN

from torchsummary import summary
from torchstat import stat
import torch
import argparse
import yaml
from addict import Dict
import os

from flopth import flopth
from ptflops import get_model_complexity_info

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='unet', choices=['unet', 'manet', 'linknet', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+', 'manet','fpn', 'segformer-b3', 'liteseg', 'stdc1', 'stdc2', 'a2fpn'], help='Choose different semantic segmention architecture')
ap.add_argument("-l", "--width", type=int, help="Width should be a multiple of 256")
ap.add_argument('-m', '--height',  type=int, help="height should be a multiple of 256")
ap.add_argument('-p', '--profiler', default='all', choices=['torchsummary', 'flopth', 'ptflops', 'stat', 'all'], help='Choose a network profiler')
args = vars(ap.parse_args())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 30
ACTIVATION = 'sigmoid' 

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
    
    CONFIG=Dict(yaml.load(open("liteseg_model/config/training.yaml"), Loader=yaml.Loader))
    model = LiteSeg.build(backbone_network,None,CONFIG,is_train=True, classes=CLASSES)
elif args["arch"] == "stdc1":
    backbone_network = "STDCNet813" # STDC1 = STDCNet813, STDC2=STDCNet1446   
    model = BiSeNet(backbone= backbone_network, n_classes=CLASSES)
elif args["arch"] == "stdc2":
    backbone_network = "STDCNet1446" # STDC1 = STDCNet813, STDC2=STDCNet1446
    model = BiSeNet(backbone= backbone_network, n_classes=CLASSES)
elif args["arch"] == "a2fpn":
    model = A2FPN(3, class_num=CLASSES)

#print(model)

width = args["width"]
height = args["height"]
print("image width: {}, height:{}".format(width, height))

input = (3, width, height)


if (args["profiler"] == "all" or args["profiler"] == "ptflops"): 
    print("=====START Profile With PTFLOPS========")
    macs, params = get_model_complexity_info(model, input, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("=====END Profile With PTFLOPS========")

if (args["profiler"] == "all" or args["profiler"] == "torchsummary"): 
    print("********START Profile With TorchSummary********")
    summary(model, input)
    print("********END Profile With PyTorchSummary*********")

if (args["profiler"] == "all" or args["profiler"] == "flopth"): 
    dummy_inputs = torch.rand(1, 3, width, height)
    print("=====START Profile With FLOPTH========")
    flops, params = flopth(model, inputs=(dummy_inputs,))
    print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
    print("=====END Profile With FLOPTH========")

if (args["profiler"] == "all" or args["profiler"] == "stat"):
    print("****START Profile With STAT****")
    stat(model.to("cpu"), input)
    print("****END Profile With STAT****")

