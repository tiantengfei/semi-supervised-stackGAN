# import torch
# import torch.nn as nn
# import torch.nn.parallel
# from torch import autograd
# import torch.nn.functional as F
#
# def upBlock(in_planes, out_planes):
#     block = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         conv3x3(in_planes, out_planes * 2),
#         nn.BatchNorm2d(out_planes * 2),
#         GLU()
#     )
#     return block
#
# def conv3x3(in_planes, out_planes):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
#                      padding=1, bias=False)
# class GLU(nn.Module):
#     def __init__(self):
#         super(GLU, self).__init__()
#
#     def forward(self, x):
#         nc = x.size(1)
#         print(nc)
#         assert nc % 2 == 0, 'channels dont divide 2!'
#         nc = int(nc / 2)
#         print("shape:{0}".format(x[:,:nc].shape))
#         return x[:, :nc] * F.sigmoid(x[:, nc:])
#
# fc = nn.Sequential(  # 32 * 16 * 4 * 4 * 2
#             nn.Linear(100, 32 * 16 * 4 * 4 * 2, bias=False),
#             nn.BatchNorm1d(32 * 16 * 4 * 4 * 2),
#             GLU())
#
# sa_1 = nn.Upsample(scale_factor=2, mode='nearest')
# upsample1 = upBlock(32 * 16, 32 * 16 / 2)
#
# input = autograd.Variable(torch.randn(20, 100))
#
#
# print(fc(input).size())
# in_code = fc(input)
# out_code = in_code.view(-1, 32 * 16, 4, 4)  # [32 * 16, 4, 4]
# out = upsample1(out_code)
# block = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         conv3x3(32 * 16, 32 * 8 * 2),
#      nn.BatchNorm2d(32 * 8 * 2),
#             GLU()
#
# )
# print("sa1:{0}".format(block(out_code).size()))
# print(out.size())
# print(fc[0])
#
# input = autograd.Variable(torch.randn(20, 32, 32, 32))
# ss = conv3x3(32, 32 * 2)
# print("image_test:{0}".format(ss(input).size()))
#
# c_code = autograd.Variable(torch.randn(20, 100))
# s_size = input.size(2)
# c_code = c_code.view(-1, 100, 1, 1)
# print(c_code.size())
# c_code = c_code.repeat(1, 1, s_size, s_size)
# print(c_code.size())


import numpy as np
import cPickle

#
# ls = [1,2,3,4]
#
# re = []
# class_num = 10
# for i in ls:
#     r = []
#     for j in range(10):
#         if j == i - 1:
#             r.append(1)
#         else:
#             r.append(0)
#
#     re.append(r)
#
#
# print(np.array(re))
#
#
# with open('/home/ttf/dataset/cifar-10-batches-py/data_batch_4', 'rb') as f:
#     dict = cPickle.load(f)
#     for key in dict.keys():
#         print(key)
#         print(dict[key][5])

#
# print(len(dict['data']))
# print(len(dict['labels']))

# from PIL import Image
# import os
#
# im = Image.open("/home/ttf/Downloads/building.jpg")
# width, height = im.size
# print("width:{0}, height:{1}".format(width, height))
#
# crop_size = 128
#
# w_size = width // crop_size
# h_size = width // crop_size
# output_dir = "out"
# if not os.path.exists("out"):
#     os.mkdir(output_dir)
# count = 0
# for i in range(w_size):
#     for j in range(h_size):
#         count += 1
#         im.crop((i * crop_size, j * crop_size,  (i+1) * crop_size, (j+1) * crop_size)). \
#            save(os.path.join(output_dir, str(crop_size) + str(count) + ".jpg"))
#         print("get {0} images".format(count))

# from PIL import Image
# import os
# import imghdr
#
# image_dir = "/home/ttf/applications/ImageNet_Utils/n02084071/cat"
# IMG_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG',
#                   'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP']
# print("total image is {0}".format(len(os.listdir(image_dir))))
# for f in os.listdir(image_dir):
#      image_type = imghdr.what(os.path.join(image_dir, f))
#      if not image_type in IMG_EXTENSIONS:
#          os.remove(os.path.join(image_dir, f))
#          print("delete {0} image".format(f))


# from PIL import Image
# import os
# building_dir = ""
# road_net_dir = ""
#
# building_images = os.listdir(building_dir).sort()
# road_net_images = os.listdir(road_net_dir).sort()
#
# assert len(building_images) == len(road_net_images), "building image number is equals to road_net image number "
#
# image_pair = zip(building_images, road_net_images)
# img_size = 256
# out_put = "combine"
# if os.path.exists(out_put):
#     os.mkdir(out_put)
#
# for img_1, img_2 in image_pair:
#     assert img_1 == img_2, "image file name should be equal"
#     images = map(Image.open, [os.path.join(building_dir, img_1), os.path.join(road_net_dir, img_2)])
#     images = [im.resize((img_size, img_size), Image.ANTIALIAS) for im in images]
#     widths, heights = zip(*(i.size for i in images))
#
#     total_width = sum(widths)
#     max_height = max(heights)
#
#     new_im = Image.new('RGB',(total_width, max_height))
#     x_offet = 0
#
#     for im in images:
#         new_im.paste(im, (x_offet, 0))
#         x_offet += im.size[0]
#
#     new_im.save(os.path.join(out_put, img_1))


# from PIL import Image
# im = Image.open("/home/ttf/applications/StackGAN-v2-master/data/imagenet/image_train/cat/zwit11.jpg").convert('RGB')
# print(im)

import torch
import torchvision
import torchvision.transforms as transforms
import sys
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', image_train=True,
#                                         download=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', image_train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# for i, data in enumerate(trainloader, 0):
#     # get the inputs
#     inputs, labels = data
#     print(labels)
#     sys.exit(0)
#
# import torchvision
# torchvision.datasets.CIFAR10
# from datasets import Cifar10Folder
#
# imsize=32
# image_transform = transforms.Compose([
#     transforms.Scale(int(imsize * 76 / 32)),
#     transforms.RandomCrop(imsize),
#     transforms.RandomHorizontalFlip()])
#
# dataset = Cifar10Folder("../data/cifar10", split_dir='image_train',
#                               base_size=32,
#                               transform=image_transform,
#                        )
#
# dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=4,  # * num_gpu,
#         drop_last=True, shuffle=True, num_workers=4)
#
# for step, data in enumerate(dataloader, 0):
#
#     print(type(data[2]))
#     print(data[2])
#     sys.exit(0)

import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch import autograd
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block



input = autograd.Variable(torch.randn(20,3,128,128))
ndf = 32

#fc = nn.Linear(3 * 32 * 32, 120)
img_code_s16 = encode_image_by_16times(ndf)
img_code_s32 = downBlock(ndf * 8, ndf * 16)
img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
img_code_s64 = downBlock(ndf * 16, ndf * 32)
img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)


uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
output = img_code_s16(input)
print(output.size())

output = img_code_s32(output)
print(output.size())
output_1 = img_code_s32_1(output)
print("output_1:{0}".format(output_1.size()))
output = img_code_s64(output)
print(output.size())

output = img_code_s64_1(output)
print(output.size())

output = img_code_s64_2(output)
print(output.size())

output = uncond_logits(output)
print(output.size())
#
# import pickle
#
# with open("../data/cifar10/image_test/cifar10_test.pkl", 'rb') as f :
#     ls = pickle.load(f)
#     print(len(ls["image_test"]))

from datasets import Cifar10Folder
label_dataset = Cifar10Folder("../data/cifar10", "cifar10_label")
label_loader = torch.utils.data.DataLoader(
        label_dataset, batch_size=64,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))


dataiter = iter(label_loader)

for i in range(200):
    a = dataiter.next()
print(a[0][1][0])


