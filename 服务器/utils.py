#!/usr/bin/env python
import argparse
import os.path as osp
import numpy as np
import logging
import time
import math
import sys
import cv2
import os
sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

def load_cam_dtu(path, num_depth=0, interval_scale=1.0):
    file = open(path)
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam


# 构造images和cams.txt的读取路径，并返回列表
def make_paths(target_path, index):
    cam_path = os.path.join(target_path,'cams')
    img_path = os.path.join(target_path,'images')

    cams_r = []
    imgs_r = []
    for i in range(len(os.listdir(img_path))):
        cams_r.append("{}.txt".format(i))
        imgs_r.append("{}.jpg".format(i))
    cams_r[0], cams_r[index] = cams_r[index], cams_r[0]
    imgs_r[0], imgs_r[index] = imgs_r[index], imgs_r[0]

    cams = [os.path.join(cam_path,c) for c in cams_r]
    imgs = [os.path.join(img_path,i) for i in imgs_r]

    return (cams,imgs)


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


# scale和crop将任意尺寸的图片转换成模型可以接受的尺寸
def scale_input(images, cams, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(len(images)):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return images, cams, depth_image


def crop_input(images, cams, base_image_size=64, height=1152, width=1600, depth_image=None):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    for view in range(len(images)):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > height:
            new_h = height
        else:
            new_h = int(math.floor(h / base_image_size) * base_image_size)
        if new_w > width:
            new_w = width
        else:
            new_w = int(math.floor(w / base_image_size) * base_image_size)
        start_h = int(math.floor((h - new_h) / 2))
        start_w = int(math.floor((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams


def norm_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 1e-7)


def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()