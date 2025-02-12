import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import transforms
import math
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
import yaml
import os

class FeatureExtractor:
    def __init__(self, config_path, which_epoch='last', gpu_ids='0', ms='1'):
        # 解析配置文件
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        self.opt = argparse.Namespace()
        for key, value in config.items():
            setattr(self.opt, key, value)
        self.opt.which_epoch = which_epoch
        self.opt.gpu_ids = gpu_ids
        self.opt.ms = ms

        str_ids = self.opt.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)

        str_ms = self.opt.ms.split(',')
        self.ms = []
        for s in str_ms:
            s_f = float(s)
            self.ms.append(math.sqrt(s_f))

        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
            cudnn.benchmark = True

        # 选择模型结构
        if self.opt.use_dense:
            model_structure = ft_net_dense(self.opt.nclasses, stride=self.opt.stride, linear_num=self.opt.linear_num)
        elif self.opt.use_NAS:
            model_structure = ft_net_NAS(self.opt.nclasses, linear_num=self.opt.linear_num)
        elif self.opt.use_swin:
            model_structure = ft_net_swin(self.opt.nclasses, linear_num=self.opt.linear_num)
        elif self.opt.use_swinv2:
            model_structure = ft_net_swinv2(self.opt.nclasses, (self.opt.h, self.opt.w), linear_num=self.opt.linear_num)
        elif self.opt.use_convnext:
            model_structure = ft_net_convnext(self.opt.nclasses, linear_num=self.opt.linear_num)
        elif self.opt.use_efficient:
            model_structure = ft_net_efficient(self.opt.nclasses, linear_num=self.opt.linear_num)
        elif self.opt.use_hr:
            model_structure = ft_net_hr(self.opt.nclasses, linear_num=self.opt.linear_num)
        else:
            model_structure = ft_net(self.opt.nclasses, stride=self.opt.stride, ibn=self.opt.ibn,
                                     linear_num=self.opt.linear_num)

        if self.opt.PCB:
            model_structure = PCB(self.opt.nclasses)

        self.model = self.load_network(model_structure)

        # Remove the final fc layer and classifier layer
        if self.opt.PCB:
            self.model = PCB_test(self.model)
        else:
            self.model.classifier.classifier = nn.Sequential()

        # Change to test mode
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = fuse_all_conv_bn(self.model)

        # 定义图像预处理转换
        if self.opt.use_swin:
            self.h, self.w = 224, 224
        else:
            self.h, self.w = 256, 128

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.opt.PCB:
            self.data_transforms = transforms.Compose([
                transforms.Resize((384, 192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.h, self.w = 384, 192

    def load_network(self, network):
        save_path = os.path.join('./model', self.opt.name, 'net_%s.pth' % self.opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            if torch.cuda.get_device_capability()[0] > 6 and len(self.opt.gpu_ids) == 1 and int(
                    torch.__version__[0]) > 1:
                print("Compiling model...")
                torch.set_float32_matmul_precision('high')
                network = torch.compile(network, mode="default", dynamic=True)
            network.load_state_dict(torch.load(save_path))
        return network

    def fliplr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(self, img):
        if self.opt.linear_num <= 0:
            if self.opt.use_swin or self.opt.use_swinv2 or self.opt.use_dense or self.opt.use_convnext:
                self.opt.linear_num = 1024
            elif self.opt.use_efficient:
                self.opt.linear_num = 1792
            elif self.opt.use_NAS:
                self.opt.linear_num = 4032
            else:
                self.opt.linear_num = 2048

        img = self.data_transforms(img).unsqueeze(0)
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, self.opt.linear_num).zero_().cuda()

        if self.opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()

        for i in range(2):
            if (i == 1):
                img = self.fliplr(img)
            input_img = Variable(img.cuda())
            for scale in self.ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                          align_corners=False)
                with torch.no_grad():
                    outputs = self.model(input_img)
                ff += outputs

        if self.opt.PCB:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        return ff.cpu().numpy().flatten()

