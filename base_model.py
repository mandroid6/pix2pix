
import os
import torch
from collections import OrderedDict
from . import networks

class BaseModel():
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def name(self):
        return 'BaseModel'
    
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudann.benchmark = True
        self.loss_name = []
        self.model_names = []
        self.visual_name = []
        self.image_paths = []
        
    def set_input(self, input):
        self.input = input
        
    