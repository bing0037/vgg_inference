
import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from vgg import *

import time

import prune_util
import yaml
import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 rew training')
parser.add_argument('--sparsity_type', type=str, default='column',
                    help ="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16_p50',
                    help ="config file name")                    
parser.add_argument('--block_size', type=int, default=16,
                    help="block size for block_wise pruning")                    
args = parser.parse_args()

# 1) Dataset:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
transform_test = transforms.Compose([transforms.ToTensor(), normalize])
cifar10_val = datasets.CIFAR10("data/cifar10", train=False, transform=transform_test, download=True)
print("dataload_cifar_val_complete...")

def validate():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 100 == 0 and i != 0:
                print("steps:", i, ", accuracy:", correct / total)

    print('Test accuracy: %f%%' % (
        100 * correct / total))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = vgg16()
# model = vgg16(pretrain=False) # training from scratch.
model = vgg16(pretrain=True) # load pretrained model.

# model = nn.DataParallel(model)
model.to(device)

batch_size = 512
data_loader = torch.utils.data.DataLoader(cifar10_val,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

# 1. Inference time of the original model:
time_original = time.time()
validate()        
print("=====> Validation time of original model: %.2f seconds!" % (time.time()-time_original))

# 2. add sparsity manually:
prune_ratios = []
with open("./profile/" + args.config_file + ".yaml", "r") as stream:
    try:
        raw_dict = yaml.load(stream)
        prune_ratios = raw_dict['prune_ratios']
    except yaml.YAMLError as exc:
        print(exc)
# print(prune_ratios)
prune_util.hard_prune(args, prune_ratios, model)

# 3. Inference time of the pruned model:
time_original2 = time.time()
validate()        
print("=====> Validation time of pruned model: %.2f seconds!" % (time.time()-time_original2))

# 4. Confirm the pruning procedure:
if args.sparsity_type == "column":
    from testers import test_column_sparsity
    test_column_sparsity(model)