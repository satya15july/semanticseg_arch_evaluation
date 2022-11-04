# USAGE: python3 training_cityscapes.py --arch deeplabv3+ --traintype parallel --epochs 50 --outpath savedModels

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt  

import time

import segmentation_models_pytorch as smp
from segtransformer import segformer_mit_b3
from liteseg_model.liteseg import LiteSeg

from dataset_cityscapes import cityscapesLoader
import config
import os
import argparse
import yaml
from addict import Dict

from utils import cross_entropy2d, get_metrics, runningScore


ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='unet', choices=['unet', 'manet', 'linknet', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+', 'manet','fpn','segformer-b3', 'liteseg'], help='Choose different semantic segmention architecture')
ap.add_argument('-t', '--traintype', default='single', choices=['single', 'parallel'], help='Choose Parallel if 2 GPUs are available')
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
ap.add_argument("-o", "--outpath", required=True,	help="Model Save path ")
args = vars(ap.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

path_data = config.CITYSCAPES_DATASET
train_epochs = args['epochs']

DISTRIBUTED_TRAINING = args['traintype']

print("Number of cuda {}".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    print("Let's use : {}".format(torch.cuda.device_count(), "GPUs!"))
    print("Let's Use PARALLEL MODE TRAINING")
    DISTRIBUTED_TRAINING = True
else:
    print("Let's Use SINGLE MODE TRAINING")
    DISTRIBUTED_TRAINING = False

## If there is a "RuntimeError: CUDA out of memory",then change the BATCH_SIZE to some lower number.
if DISTRIBUTED_TRAINING:
    BATCH_SIZE = config.BATCH_SIZE
else:
    BATCH_SIZE = 4 # Change this number based on your System capacity.Normally,11GB Graphics card this is fine.

OUTPUT_PATH = args['outpath']

train_data = cityscapesLoader(
    root = path_data, 
    split='train'
    )

val_data = cityscapesLoader(
    root = path_data, 
    split='val'
    )

train_loader = DataLoader(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers = config.NUM_WORKERS,
    #pin_memory = pin_memory  # gave no significant advantage
)

val_loader = DataLoader(
    val_data,
    batch_size = BATCH_SIZE,
    num_workers = config.NUM_WORKERS,
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

if DISTRIBUTED_TRAINING:
    model = nn.DataParallel(model)
    model = model.cuda()
else:
    model = model.to(config.DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


def train(train_loader, model, optimizer, epoch_i, epoch_total):
        count = 0
        
        # List to cumulate loss during iterations
        loss_list = []
        for (images, labels) in train_loader:
            count += 1
            
            # we used model.eval() below. This is to bring model back to training mood.
            model.train()
            #pred = None
            if DISTRIBUTED_TRAINING:
                images = images.cuda()
                labels = labels.cuda()
                # Model Prediction
                #pred = model(images).cuda()
            else:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                #pred = model(images).to(config.DEVICE)

            pred = model(images)
            # Loss Calculation
            loss = cross_entropy2d(pred, labels)
            loss_list.append(loss)

            # optimiser
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # interval to print train statistics
            if count % 50 == 0:
                fmt_str = "Image: {:d} in epoch: [{:d}/{:d}]  and Loss: {:.4f}"
                print_str = fmt_str.format(
                    count,
                    epoch_i + 1,
                    epoch_total,
                    loss.item()
                )
                print(print_str)
                   
#           # break for testing purpose
#             if count == 10:
#                 break
        return(loss_list)

def save_model(network, epoch_label):
    print("save model: epoch_label = {}".format(epoch_label))
    save_filename = '{}_{}.pth'.format(args['arch'], epoch_label)
    save_path = os.path.join('./savedModels', save_filename)
    if DISTRIBUTED_TRAINING:
        torch.save(network.module.state_dict(), save_path)
    else:
        torch.save(network.state_dict(), save_path)

def validate(val_loader, model, epoch_i):
    
    # tldr: to make layers behave differently during inference (vs training)
    model.eval()
    
    # enable calculation of confusion matrix for n_classes = 19
    running_metrics_val = runningScore(19)
    
    # empty list to add Accuracy and Jaccard Score Calculations
    acc_sh = []
    js_sh = []
    
    with torch.no_grad():
        for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):
            if DISTRIBUTED_TRAINING:
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()
            else:
                val_images = val_images.to(config.DEVICE)
                val_labels = val_labels.to(config.DEVICE)
            
            # Model prediction
            val_pred = model(val_images)
            
            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes
            pred = val_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()
            
            # Updating Mertics
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])
            accuracy = sh_metrics[0]
            print("sh_metrics[0]: {}".format(sh_metrics[0]))
            print("epoch_i: {}".format(epoch_i))
            print("config.best_accuracy: {}".format(config.best_accuracy))
            if (epoch_i %10 == 0 and accuracy > config.best_accuracy):
                config.best_accuracy = accuracy

                save_model(model, epoch_i)
#            # break for testing purpose
#             if image_num == 10:
#                 break                

    score = running_metrics_val.get_scores()
    running_metrics_val.reset()
    
    acc_s = sum(acc_sh)/len(acc_sh)
    js_s = sum(js_sh)/len(js_sh)
    score["acc"] = acc_s
    score["js"] = js_s
    
    print("Different Metrics were: ", score)  
    return(score)

if __name__ == "__main__":

    # to hold loss values after each epoch
    loss_all_epochs = []
    
    # to hold different metrics after each epoch
    Specificity_ = []
    Senstivity_ = []
    F1_ = []
    acc_ = []
    js_ = []
    
    for epoch_i in range(train_epochs):
        # training
        print(f"Epoch {epoch_i + 1}\n-------------------------------")
        t1 = time.time()
        loss_i = train(train_loader, model, optimizer, epoch_i, train_epochs)
        loss_all_epochs.append(loss_i)
        t2 = time.time()
        print("It took: ", t2-t1, " unit time")

        # metrics calculation on validation data
        dummy_list = validate(val_loader, model, epoch_i)   
        
        # Add metrics to empty list above
        Specificity_.append(dummy_list["Specificity"])
        Senstivity_.append(dummy_list["Senstivity"])
        F1_.append(dummy_list["F1"])
        acc_.append(dummy_list["acc"])
        js_.append(dummy_list["js"])

    # loss_all_epochs: contains 2d list of tensors with: (epoch, loss tensor)
    # converting to 1d list for plotting
    loss_1d_list = [item for sublist in loss_all_epochs for item in sublist]
    loss_list_numpy = []
    for i in range(len(loss_1d_list)):
        z = loss_1d_list[i].cpu().detach().numpy()
        loss_list_numpy.append(z)
    plt.xlabel("Images used in training epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(loss_list_numpy)
    plt.show()

    plt.clf()

    x = [i for i in range(1, train_epochs + 1)]

    # plot 5 metrics: Specificity, Senstivity, F1 Score, Accuracy, Jaccard Score
    plt.plot(x,Specificity_, label='Specificity')
    plt.plot(x,Senstivity_, label='Senstivity')
    plt.plot(x,F1_, label='F1 Score')
    plt.plot(x,acc_, label='Accuracy')
    plt.plot(x,js_, label='Jaccard Score')

    plt.grid(linestyle = '--', linewidth = 0.5)

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # tldr: to make layers behave differently during inference (vs training)
    model.eval()

    with torch.no_grad():
        for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):

            if DISTRIBUTED_TRAINING:
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()
            else:
                val_images = val_images.to(config.DEVICE)
                val_labels = val_labels.to(config.DEVICE)
        
            # model prediction
            val_pred = model(val_images)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes        
            prediction = val_pred.data.max(1)[1].cpu().numpy()
            ground_truth = val_labels.data.cpu().numpy()

            # replace 100 to change number of images to print. 
            # 500 % 100 = 5. So, we will get 5 predictions and ground truths
            if image_num % 100 == 0:
            
                # Model Prediction
                decoded_pred = val_data.decode_segmap(prediction[0])
                plt.imshow(decoded_pred)
                plt.show()
                plt.clf()
            
                # Ground Truth
                decode_gt = val_data.decode_segmap(ground_truth[0])
                plt.imshow(decode_gt)
                plt.show()
