"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import numpy as np
import torch
import cv2
from vggNet import *


def vgg_1_1(img_folder):
    # specify gpu id
    gpu_id = 0

    model = vgg16(pretrained=True)
    model.eval()
    model.cuda(gpu_id)

    img_path = img_folder
    image = cv2.imread(img_path)
    h, w, ch = image.shape

    # 对图像内容计算
    input1 = image.transpose((2, 0, 1))
    input1 = np.float32(input1) / 255.0
    input1 = np.reshape(input1, [1, ch, h, w])
    input1 = torch.from_numpy(input1)
    input1 = input1.cuda(gpu_id)

    out = model(input1)
    return  out


if __name__ == '__main__':
    print(vgg_1_1("D:\Coding\Python\PycharmProjects\SEAL\data\input/28083.jpg"))
