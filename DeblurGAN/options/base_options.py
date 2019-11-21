import argparse
import os
from util import util
import torch
import numpy as np

class BaseConfiguration:
    	def __init__(self):
    		self.initialized = False
    		
		def initialize(self):
			self.dataroot = "D:\Photos\TrainingData\BlurredSharp\combined"
			self.batchSize = 1
			self.loadSizeX = 640
			self.loadSizeY = 360
			self.fineSize = 256
			self.input_nc = 3
			self.output_nc = 3
			self.ngf = 64
			self.ndf = 64
			self.which_model_netD = 'basic'
			self.which_model_netG = 'resnet_9blocks'
			self.learn_residual = True
			self.gan_type = 'wgan-gp'
			self.n_layers_D = 3
			self.gpu_ids = -1 # 0, 1, 2 GPU
			self.name = 'experiment_name'
			self.dataset_mode = 'aligned'
			self.model = 'content_gan'
			self.which_direction = 'AtoB'
			self.nThreads = 2
			self.checkpoints_dir = '../neural_networks/Deblur' # neural networks
			self.norm = 'instance'
			self.serial_batches = 'store_true',
			self.display_winsize = 256
			self.display_id = 1
			self.display_port = 8097
			self.display_single_pane_ncols = 0
			self.no_dropout = 'store_true'
			self.max_dataset_size = np.inf
			self.resize_or_crop = 'resize_and_crop'
			self.no_flip = 'store_true'
			self.initialize = True

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False



	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.isTrain = self.isTrain   # train or test

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		args = vars(self.opt)

		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		util.mkdirs(expr_dir)
		
		return self.opt
