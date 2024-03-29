import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import json
import zipfile
from collections import OrderedDict


from .configure_craft_pytorch import ConfigureCRAFTPytorch
from .imgproc import ImgProc
from .craft import CRAFT
from .file_utils import FileUtils
from .craft_utils import CraftUtils


class CRAFTTextDetector:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.args = ConfigureCRAFTPytorch(
            trained_model= '../neural_networks/CRAFT/craft_ic15_20k.pth',
            test_folder='books_images')
        """ For test images in a folder """
        self.file_utils = FileUtils()
        self.craft_utils = CraftUtils()
        self.image_list, _, _ = self.file_utils.get_files(self.args.test_folder)
        self.imgproc = ImgProc()

        self.result_folder = './result/'
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

    def get_network(self):
        net = CRAFT()

        print('Loading weights from checkpoint (' + self.args.trained_model + ')')
        if torch.cuda.device_count() > 0:
            net.load_state_dict(self.copyStateDict(torch.load(self.args.trained_model)))
        else:
            net.load_state_dict(self.copyStateDict(torch.load(self.args.trained_model, map_location='cpu')))

        if torch.cuda.device_count() > 0:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()
        return net

    def detect_one(self, image_path=None):
        net = self.get_network()
        t = time.time()
        print("Test image: {:s}".format(image_path))
        image = self.imgproc.loadImage(image_path)
        boxes, polys, score_text = self.test_net(net, image, self.args.text_threshold, self.args.link_threshold, self.args.low_text, torch.cuda.device_count() > 0, self.args.poly, None)
        print("elapsed time : {}s".format(time.time() - t))
        return boxes, image


    def detect(self, image_path=None):
        self.image_path = image_path
            # load net
        net = CRAFT()     # initialize

        print('Loading weights from checkpoint (' + self.args.trained_model + ')')
        if self.args.cuda:
            net.load_state_dict(self.copyStateDict(torch.load(self.args.trained_model)))
        else:
            net.load_state_dict(self.copyStateDict(torch.load(self.args.trained_model, map_location='cpu')))

        if self.args.cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        # LinkRefiner
        refine_net = None
        if self.args.refine:
            from refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.args.refiner_model + ')')
            if self.args.cuda:
                refine_net.load_state_dict(self.copyStateDict(torch.load(self.args.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(self.copyStateDict(torch.load(self.args.refiner_model, map_location='cpu')))

            refine_net.eval()
            self.args.poly = True

        t = time.time()

        # load data
        for k, image_path in enumerate(self.image_list):
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(self.image_list), image_path))
            image = self.imgproc.loadImage(image_path)

            bboxes, polys, score_text = self.test_net(net, image, self.args.text_threshold, self.args.link_threshold, self.args.low_text, self.args.cuda, self.args.poly, refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = self.result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            self.file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=self.result_folder)

        print("elapsed time : {}s".format(time.time() - t))

    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = self.imgproc.resize_aspect_ratio(image, self.args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = self.imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = self.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = self.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = self.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = self.imgproc.cvt2HeatmapImg(render_img)

        if self.args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text
