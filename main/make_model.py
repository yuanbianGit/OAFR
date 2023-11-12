import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from model.backbones.swin_transformer_pad import SwinTransformer_pad
from model.backbones.swin_transformer import SwinTransformer

import random
import numpy as np
from einops import einops
from model.backbones.resnet_nl import ResNetNL
from model.backbones.gem_pool import GeneralizedMeanPoolingP

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, occ_agu=False,  personOCC_pro = 0.5,label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        side_L = int(x.size(3))
        '''这个地方的长宽要改！'''
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1 - personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(3, 6)
                h_start = h_index * 32
                h_end = 224

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / 32)] = 0

                occ_info = torch.zeros(side_L)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / (side_L / 7)
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 6)  # 0 和6 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4 - h_index)
                else:
                    occ_parNum = random.randint(1, 7 - h_index)

                h_start = h_index * 32
                h_end = (h_index + occ_parNum) * 32

                h_start_copy = h_start + 3 * 32
                h_end_copy = h_end + 3 * 32

                if h_start_copy < x.size(3):
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :,
                                                h_start_copy - int(x.size(3)):h_end_copy - int(x.size(3)), :]
                    from_copy_tensor_label[
                    int((h_start_copy - int(x.size(3))) / 32): int((h_end_copy - int(x.size(3))) / 32)] = 0

                occ_info = torch.zeros(side_L)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / (side_L / 7)

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)
        # x = self.base(x)
        # global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, occ_agu=False, personOCC_pro=0.5,label=None, cam_label= None, view_label=None):
        side_H = int(x.size(2))
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1 - personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                part_L = int(side_H / 8)
                h_index = random.randint(3, 7)
                h_start = h_index * part_L
                h_end = side_H

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / part_L)] = 0

                occ_info = torch.zeros(side_H)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 7)  # 0 和7 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4 - h_index)
                else:
                    occ_parNum = random.randint(1, 8 - h_index)

                part_L = int(side_H / 8)
                h_start = h_index * part_L
                h_end = (h_index + occ_parNum) * part_L

                h_start_copy = h_start + 4 * part_L
                h_end_copy = h_end + 4 * part_L

                if h_start_copy < side_H:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy - side_H:h_end_copy - side_H, :]
                    from_copy_tensor_label[int((h_start_copy - side_H) / 32): int((h_end_copy - side_H) / 32)] = 0

                occ_info = torch.zeros(side_H)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)

        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vit_oafr_224(nn.Module):
    def __init__(self, clas_nums, camera_num, view_num, cfg, factory):
        super(build_vit_oafr_224, self).__init__()

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.num_classes = clas_nums

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.occ_predict1 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.occ_predict2 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict3 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict4 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict5 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict6 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict7 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.hen = torch.nn.AdaptiveAvgPool2d((7, 1))
        # self.hen = torch.nn.AdaptiveMaxPool1d(7)

        self.occ_pooling = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x,occ_agu=False, personOCC_pro=0.5, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # import time
        # start_time = time.time()  # 程序开始时间
        # label要为原始label的两倍！
        # 遮挡标签为0，未被遮挡标签为1
        side_H = int(x.size(2))
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1 - personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                part_L = int(side_H / 7)
                h_index = random.randint(3, 6)
                h_start = h_index * part_L
                h_end = side_H

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / part_L)] = 0

                occ_info = torch.zeros(side_H)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 6)  # 0 和6 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4 - h_index)
                else:
                    occ_parNum = random.randint(1, 7 - h_index)

                part_L = int(side_H / 7)
                h_start = h_index * part_L
                h_end = (h_index + occ_parNum) * part_L

                h_start_copy = h_start + 3 * part_L
                h_end_copy = h_end + 3 * part_L

                if h_start_copy < x.size(3):
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :,
                                                h_start_copy - int(x.size(3)):h_end_copy - int(x.size(3)), :]
                    from_copy_tensor_label[
                    int((h_start_copy - int(x.size(3))) / 32): int((h_end_copy - int(x.size(3))) / 32)] = 0

                occ_info = torch.zeros(side_H)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)

        features = self.base(x, cam_label=cam_label, view_label=view_label)
        b1_feat = self.b1(features)  # [64, 129, 768]
        # global_feat = b1_feat[:, 0]
        feat_map_cls = b1_feat[:, 1:]

        if self.training:
            '''要新加的loss加在这里'''

            # occ_lable_h = occ_label[:,0]
            # where = torch.not_equal(occ_lable_h, 1)
            # index_occ = torch.where(where)
            # h_num = torch.sum(1-occ_lable_h)
            # occ_feat = torch.zeros((int(feat_map_cls.size(0)/2),int(feat_map_cls.size(1)),int(h_num.item()),7)).to('cuda')
            # occ_feat[:,:,:,:] = feat_map_cls[int(feat_map_cls.size(0)/2):,:, index_occ[0].min().item():index_occ[0].max().item()+1 ,:]
            # occ_feat = self.occ_pooling(occ_feat).squeeze()
            occ_feat = None

        # 这里分离了occ_pre的梯度！
        feat_map_cls = einops.rearrange(feat_map_cls, 'b (h w) l -> b l h w', h=14)
        # global_feat = self.gap(feat_map_cls).squeeze()
        # feat_map = feat_map_cls.detach()  # (B,C,L)

        global_feat = self.gap(feat_map_cls).squeeze(dim=3)
        global_feat = global_feat.squeeze(dim=2)
        feat_map = feat_map_cls.detach()

        h_feat = self.hen(feat_map).squeeze(dim=-1)  # B, C, 7
        h_local_0 = h_feat[..., 0]
        h_local_1 = h_feat[..., 1]
        h_local_2 = h_feat[..., 2]
        h_local_3 = h_feat[..., 3]
        h_local_4 = h_feat[..., 4]
        h_local_5 = h_feat[..., 5]
        h_local_6 = h_feat[..., 6]
        occ_pre_label = []
        occ_pre_label.append(self.occ_predict1(h_local_0))
        occ_pre_label.append(self.occ_predict2(h_local_1))
        occ_pre_label.append(self.occ_predict3(h_local_2))
        occ_pre_label.append(self.occ_predict4(h_local_3))
        occ_pre_label.append(self.occ_predict5(h_local_4))
        occ_pre_label.append(self.occ_predict6(h_local_5))
        occ_pre_label.append(self.occ_predict7(h_local_6))
        occ_pre_label = torch.cat(occ_pre_label, dim=0)

        '''用标签0,1做'''
        co_h = torch.softmax(occ_pre_label, dim=1)
        co_h_pro = co_h[:,1].view(7,-1)
        co_h = co_h.argmax(dim=1)  # (7B)
        # # (7B,2)；7B是batch里每个part（B,2）cat起来的！所以是七个局部的label* B叠起来
        co_h = co_h.view(7, -1)

        h_feat_cls = self.hen(feat_map_cls).squeeze(dim=-1)
        h_local_cls_0 = h_feat_cls[..., 0]
        h_local_cls_1 = h_feat_cls[..., 1]
        h_local_cls_2 = h_feat_cls[..., 2]
        h_local_cls_3 = h_feat_cls[..., 3]
        h_local_cls_4 = h_feat_cls[..., 4]
        h_local_cls_5 = h_feat_cls[..., 5]
        h_local_cls_6 = h_feat_cls[..., 6]


        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        # end_time = time.time()  # 程序结束时间
        # run_time = end_time - start_time  # 程序的运行时间，单位为秒
        # print(run_time)
        if self.training:

            cls_score = self.classifier(feat)
            return cls_score, global_feat, occ_feat, occ_lable_gt_2d,occ_lable_gt, occ_pre_label,torch.cat(
                [h_local_cls_0 / 7, h_local_cls_1 / 7, h_local_cls_2 / 7, h_local_cls_3 / 7, h_local_cls_4 / 7, h_local_cls_5 / 7,
                 h_local_cls_6 / 7]).reshape(7, -1, 768), rand_index, from_copy_tensor_label
        else:

            return global_feat, torch.cat(
                [h_local_cls_0 / 7, h_local_cls_1 / 7, h_local_cls_2 / 7, h_local_cls_3 / 7, h_local_cls_4 / 7, h_local_cls_5 / 7,
                 h_local_cls_6 / 7]).reshape(7, -1, 768), co_h, co_h_pro

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vit_oafr_256(nn.Module):
    def __init__(self, clas_nums, camera_num, view_num, cfg, factory):
        super(build_vit_oafr_256, self).__init__()

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.num_classes = clas_nums

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.occ_predict1 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.occ_predict2 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict3 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict4 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict5 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict6 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict7 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.occ_predict8 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.hen = torch.nn.AdaptiveAvgPool2d((8, 1))
        # self.hen = torch.nn.AdaptiveMaxPool1d(7)
        self.occ_pooling = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x,occ_agu=False, personOCC_pro=0.5, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # label要为原始label的两倍！
        # 遮挡标签为0，未被遮挡标签为1
        side_H = int(x.size(2))
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1 - personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                part_L = int(side_H / 8)
                h_index = random.randint(3, 7)
                h_start = h_index * part_L
                h_end = side_H

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / part_L)] = 0

                occ_info = torch.zeros(side_H)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])

                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 7)  # 0 和7 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4 - h_index)
                else:
                    occ_parNum = random.randint(1, 8 - h_index)

                part_L = int(side_H / 8)
                h_start = h_index * part_L
                h_end = (h_index + occ_parNum) * part_L

                h_start_copy = h_start + 4 * part_L
                h_end_copy = h_end + 4 * part_L

                if h_start_copy < side_H:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy - side_H:h_end_copy - side_H, :]
                    from_copy_tensor_label[int((h_start_copy - side_H) / 32): int((h_end_copy - side_H) / 32)] = 0

                occ_info = torch.zeros(side_H)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)

        features = self.base(x, cam_label=cam_label, view_label=view_label)
        b1_feat = self.b1(features)  # [64, 129, 768]
        # global_feat = b1_feat[:, 0]
        feat_map_cls = b1_feat[:, 1:]

        if self.training:
            '''要新加的loss加在这里'''

            # occ_lable_h = occ_label[:,0]
            # where = torch.not_equal(occ_lable_h, 1)
            # index_occ = torch.where(where)
            # h_num = torch.sum(1-occ_lable_h)
            # occ_feat = torch.zeros((int(feat_map_cls.size(0)/2),int(feat_map_cls.size(1)),int(h_num.item()),7)).to('cuda')
            # occ_feat[:,:,:,:] = feat_map_cls[int(feat_map_cls.size(0)/2):,:, index_occ[0].min().item():index_occ[0].max().item()+1 ,:]
            # occ_feat = self.occ_pooling(occ_feat).squeeze()
            occ_feat = None

        # 这里分离了occ_pre的梯度！
        print(feat_map_cls.size())
        feat_map_cls = einops.rearrange(feat_map_cls, 'b (h w) l -> b l h w', h=16)
        global_feat = self.gap(feat_map_cls).squeeze(dim=3)
        global_feat = global_feat.squeeze(dim=2)
        feat_map = feat_map_cls.detach()

        h_feat = self.hen(feat_map).squeeze(dim=-1)  # B, C, 7
        h_local_0 = h_feat[..., 0]
        h_local_1 = h_feat[..., 1]
        h_local_2 = h_feat[..., 2]
        h_local_3 = h_feat[..., 3]
        h_local_4 = h_feat[..., 4]
        h_local_5 = h_feat[..., 5]
        h_local_6 = h_feat[..., 6]
        h_local_7 = h_feat[..., 7]
        occ_pre_label = []
        occ_pre_label.append(self.occ_predict1(h_local_0))
        occ_pre_label.append(self.occ_predict2(h_local_1))
        occ_pre_label.append(self.occ_predict3(h_local_2))
        occ_pre_label.append(self.occ_predict4(h_local_3))
        occ_pre_label.append(self.occ_predict5(h_local_4))
        occ_pre_label.append(self.occ_predict6(h_local_5))
        occ_pre_label.append(self.occ_predict7(h_local_6))
        occ_pre_label.append(self.occ_predict8(h_local_7))
        occ_pre_label = torch.cat(occ_pre_label, dim=0)

        '''用标签0,1做'''
        co_h = torch.softmax(occ_pre_label, dim=1)
        co_h_pro = co_h[:,1].view(8, -1)
        co_h = co_h.argmax(dim=1)  # (7B)
        # # (7B,2)；7B是batch里每个part（B,2）cat起来的！所以是七个局部的label* B叠起来
        co_h = co_h.view(8, -1)

        h_feat_cls = self.hen(feat_map_cls).squeeze(dim=-1)
        h_local_cls_0 = h_feat_cls[..., 0]
        h_local_cls_1 = h_feat_cls[..., 1]
        h_local_cls_2 = h_feat_cls[..., 2]
        h_local_cls_3 = h_feat_cls[..., 3]
        h_local_cls_4 = h_feat_cls[..., 4]
        h_local_cls_5 = h_feat_cls[..., 5]
        h_local_cls_6 = h_feat_cls[..., 6]
        h_local_cls_7 = h_feat_cls[..., 7]
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:

            cls_score = self.classifier(feat)
            return cls_score, global_feat, occ_feat, occ_lable_gt_2d, occ_lable_gt, occ_pre_label, torch.cat(
                [h_local_cls_0 / 8, h_local_cls_1 / 8, h_local_cls_2 / 8, h_local_cls_3 / 8, h_local_cls_4 / 8,
                 h_local_cls_5 / 8,
                 h_local_cls_6 / 8, h_local_cls_7 / 8]).reshape(8, -1, 768), rand_index, from_copy_tensor_label
        else:
            return global_feat, torch.cat(
                [h_local_cls_0 / 8, h_local_cls_1 / 8, h_local_cls_2 / 8, h_local_cls_3 / 8, h_local_cls_4 / 8,
                 h_local_cls_5 / 8,
                 h_local_cls_6 / 8, h_local_cls_7 / 8]).reshape(8, -1, 768), co_h, co_h_pro

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_swin(nn.Module):
    def __init__(self, clas_nums, camera_num, view_num, cfg, factory):
        super(build_swin, self).__init__()
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.classes = clas_nums
        self.in_planes = 1024
        self.base = SwinTransformer()
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH)['model'])
            print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))

        self.classifier = nn.Linear(self.in_planes, self.classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x,occ_agu=False, personOCC_pro=0.5, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        global_feat = self.base.forward_features(x)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_swin_oafr(nn.Module):

    def __init__(self, clas_nums, camera_num, view_num, cfg, factory):
        super(build_swin_oafr, self).__init__()

        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.classes = clas_nums
        self.in_planes = 1024

        if cfg.MODEL.PRETRAIN_PATH=='../pretrainedModel/swin_base_patch4_window7_224.pth':
            self.base = SwinTransformer(num_classes=1000)
        else:
            self.base = SwinTransformer(num_classes=21841)

        #
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH)['model'])
            print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))

        self.classifier = nn.Linear(self.in_planes, self.classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # <editor-fold desc="occ_predicts">
        self.occ_predict1 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.occ_predict2 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict3 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict4 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict5 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict6 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict7 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        # </editor-fold>
        self.hen = torch.nn.AdaptiveAvgPool2d((7, 1))
        self.occ_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.ptp_m = cfg.SOLVER.PERSON_m
    def forward(self, x,occ_agu=False, personOCC_pro=0.5, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # label要为原始label的两倍！
        # 遮挡标签为0，未被遮挡标签为1
        side_L = int(x.size(3))
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1-personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4],b_index[2::4],b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4],a_index[2::4],a_index[3::4]])
                index = np.concatenate([a,b])


                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                # h_index = random.randint(3, 6)
                h_index = random.randint(self.ptp_m, 6)
                h_start = h_index * 32
                h_end = 224

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / 32)] = 0

                occ_info = torch.zeros(side_L)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / (side_L / 7)
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((7, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(7).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])


                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 6)  # 0 和6 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4-h_index)
                else:
                    occ_parNum = random.randint(1, 7-h_index)

                h_start = h_index * 32
                h_end = (h_index + occ_parNum) * 32

                h_start_copy = h_start + 3*32
                h_end_copy = h_end + 3*32

                if h_start_copy < x.size(3):
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy - int(x.size(3)):h_end_copy - int(x.size(3)), :]
                    from_copy_tensor_label[int((h_start_copy - int(x.size(3))) / 32): int((h_end_copy - int(x.size(3))) / 32)] = 0

                occ_info = torch.zeros(side_L)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(7)
                # print(occ_info)
                occ_label = torch.zeros(7)
                for i in range(7):
                    occ_label[i] = torch.sum(occ_info[i]) / (side_L / 7)

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)
        x = self.base.patch_embed(x)
        if self.base.ape:
            x = x + self.base.absolute_pos_embed
        x = self.base.pos_drop(x)
        for i in range(0, len(self.base.layers)):
            x = self.base.layers[i](x)

        x = self.base.norm(x)  # B L C
        # 这里写成转为B,C,H,W的形状！ 然后输出！
        # print(x.size())
        feat_map_cls = einops.rearrange(x, 'b (h w) l -> b l h w', h=7)

        if self.training:
            '''要新加的loss加在这里'''

            # occ_lable_h = occ_label[:,0]
            # where = torch.not_equal(occ_lable_h, 1)
            # index_occ = torch.where(where)
            # h_num = torch.sum(1-occ_lable_h)
            # occ_feat = torch.zeros((int(feat_map_cls.size(0)/2),int(feat_map_cls.size(1)),int(h_num.item()),7)).to('cuda')
            # occ_feat[:,:,:,:] = feat_map_cls[int(feat_map_cls.size(0)/2):,:, index_occ[0].min().item():index_occ[0].max().item()+1 ,:]
            # occ_feat = self.occ_pooling(occ_feat).squeeze()
            occ_feat = None

        # 这里分离了occ_pre的梯度！
        feat_map = feat_map_cls.detach()
        x = self.base.avgpool(x.transpose(1, 2))  # B C 1
        global_feat = torch.flatten(x, 1)  ## B C
        h_feat = self.hen(feat_map).squeeze(dim=-1)  # B, C, 7

        h_local_0 = h_feat[..., 0]
        h_local_1 = h_feat[..., 1]
        h_local_2 = h_feat[..., 2]
        h_local_3 = h_feat[..., 3]
        h_local_4 = h_feat[..., 4]
        h_local_5 = h_feat[..., 5]
        h_local_6 = h_feat[..., 6]

        occ_pre_label = []
        occ_pre_label.append(self.occ_predict1(h_local_0))
        occ_pre_label.append(self.occ_predict2(h_local_1))
        occ_pre_label.append(self.occ_predict3(h_local_2))
        occ_pre_label.append(self.occ_predict4(h_local_3))
        occ_pre_label.append(self.occ_predict5(h_local_4))
        occ_pre_label.append(self.occ_predict6(h_local_5))
        occ_pre_label.append(self.occ_predict7(h_local_6))
        occ_pre_label = torch.cat(occ_pre_label, dim=0)

        '''用标签0,1做'''
        co_h = torch.softmax(occ_pre_label, dim=1)
        co_h_pro = co_h[:,1].view(7,-1)
        co_h = co_h.argmax(dim=1)  # (7B)
        # # (7B,2)；7B是batch里每个part（B,2）cat起来的！所以是七个局部的label* B叠起来
        co_h = co_h.view(7, -1)

        h_feat_cls = self.hen(feat_map_cls).squeeze(dim=-1)
        h_local_cls_0 = h_feat_cls[..., 0]
        h_local_cls_1 = h_feat_cls[..., 1]
        h_local_cls_2 = h_feat_cls[..., 2]
        h_local_cls_3 = h_feat_cls[..., 3]
        h_local_cls_4 = h_feat_cls[..., 4]
        h_local_cls_5 = h_feat_cls[..., 5]
        h_local_cls_6 = h_feat_cls[..., 6]

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:

            cls_score = self.classifier(feat)
            return cls_score, global_feat, occ_feat, occ_lable_gt_2d,occ_lable_gt, occ_pre_label,torch.cat(
                [h_local_cls_0 / 7, h_local_cls_1 / 7, h_local_cls_2 / 7, h_local_cls_3 / 7, h_local_cls_4 / 7, h_local_cls_5 / 7,
                 h_local_cls_6 / 7]).reshape(7, -1, 1024), rand_index, from_copy_tensor_label
        else:
            return global_feat, torch.cat(
                [h_local_cls_0 / 7, h_local_cls_1 / 7, h_local_cls_2 / 7, h_local_cls_3 / 7, h_local_cls_4 / 7, h_local_cls_5 / 7,
                 h_local_cls_6 / 7]).reshape(7, -1, 1024), co_h, co_h_pro

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_swin_oafr_256(nn.Module):

    def __init__(self, clas_nums, camera_num, view_num, cfg, factory):
        super(build_swin_oafr_256, self).__init__()

        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = clas_nums
        self.in_planes = 1024

        self.base = SwinTransformer_pad(num_classes=21841)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH)['model'],strict=True)
            print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))

        #
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # <editor-fold>
        self.occ_predict1 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.occ_predict2 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict3 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict4 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict5 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict6 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict7 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.occ_predict8 = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        # </editor-fold>
        self.hen = torch.nn.AdaptiveAvgPool2d((8, 1))
        self.occ_pooling = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x,occ_agu=False, personOCC_pro=0.5, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # label要为原始label的两倍！
        # 遮挡标签为0，未被遮挡标签为1
        side_H = int(x.size(2))
        if occ_agu and self.training:
            pro = random.random()
            if pro > (1-personOCC_pro):
                # 用另一副图的从上而下去做！模拟行人遮挡
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                '''之前的生成随机的方式！
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4],b_index[2::4],b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4],a_index[2::4],a_index[3::4]])
                index = np.concatenate([a,b])


                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                part_L = int(side_H/8)
                h_index = random.randint(3, 7)
                h_start = h_index * part_L
                h_end = side_H

                x[:, :, h_start:h_end, :] = copy_tensor[:, :, 0:h_end - h_start, :]
                from_copy_tensor_label[0:int((h_end - h_start) / part_L)] = 0

                occ_info = torch.zeros(side_H)
                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L
                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')

                occ_label = torch.where(occ_label > 0.7, zero, one)
                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()
                # 为1是扣出来的图！

                x = torch.cat([x_ori, x], dim=0)
            else:
                # 用别人的部分去roll到1/2下去做！
                '''之前的生成乱序的方法
                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                np.random.shuffle(index[0:int(x.size(0) / 2)])
                np.random.shuffle(index[int(x.size(0) / 2):])
                '''
                x_ori = x.clone()
                occ_label_ori = torch.ones((8, x.size(0))).to('cuda')
                from_copy_tensor_label = torch.ones(8).to('cuda')

                index = np.array(range(int(x.size(0))))
                index = np.roll(index, int(x.size(0) / 2))
                a_index = index[:int(x.size(0) / 2)]
                b_index = index[int(x.size(0) / 2):]
                b_index = np.roll(b_index, random.randint(1, int(x.size(0) / 2)))
                a_index = np.roll(a_index, random.randint(1, int(x.size(0) / 2)))
                b = np.concatenate([b_index[0::4], b_index[1::4], b_index[2::4], b_index[3::4]])
                a = np.concatenate([a_index[0::4], a_index[1::4], a_index[2::4], a_index[3::4]])
                index = np.concatenate([a, b])


                # feat_occ label
                copy_tensor = torch.zeros_like(x)
                rand_index = torch.tensor(index).to('cuda')
                copy_tensor[:, :, :, :] = x[rand_index]

                h_index = random.randint(0, 7)  # 0 和7 都有可能
                if h_index < 4:
                    occ_parNum = random.randint(1, 4-h_index)
                else:
                    occ_parNum = random.randint(1, 8-h_index)

                part_L = int(side_H / 8)
                h_start = h_index * part_L
                h_end = (h_index + occ_parNum) * part_L

                h_start_copy = h_start + 4*part_L
                h_end_copy = h_end + 4*part_L

                if h_start_copy < side_H:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy:h_end_copy, :]
                    from_copy_tensor_label[int(h_start_copy / 32): int(h_end_copy / 32)] = 0
                else:
                    x[:, :, h_start:h_end, :] = copy_tensor[:, :, h_start_copy - side_H:h_end_copy - side_H, :]
                    from_copy_tensor_label[int((h_start_copy - side_H) / 32): int((h_end_copy - side_H) / 32)] = 0

                occ_info = torch.zeros(side_H)

                occ_info[h_start:h_end] = 1
                occ_info = occ_info.chunk(8)
                # print(occ_info)
                occ_label = torch.zeros(8)
                for i in range(8):
                    occ_label[i] = torch.sum(occ_info[i]) / part_L

                occ_label = occ_label.unsqueeze(dim=1).repeat((1, x.size(0))).to('cuda')  # （7,B）

                one = torch.ones_like(occ_label).to('cuda')
                zero = torch.zeros_like(occ_label).to('cuda')
                occ_label = torch.where(occ_label > 0.7, zero, one)  # 代表全被1占了才能叫被占！！

                occ_lable_gt_2d = torch.cat([occ_label_ori, occ_label], dim=1).long()  # (7,2B)
                occ_lable_gt = occ_lable_gt_2d.flatten()

                x = torch.cat([x_ori, x], dim=0)
        x = self.base.patch_embed(x)
        if self.base.ape:
            x = x + self.base.absolute_pos_embed
        x = self.base.pos_drop(x)
        for i in range(0, len(self.base.layers)):
            x = self.base.layers[i](x)

        x = self.base.norm(x)  # B L C
        # 这里写成转为B,C,H,W的形状！ 然后输出！
        # print(x.size())
        feat_map_cls = einops.rearrange(x, 'b (h w) l -> b l h w', h=8)
        # print(x.size())
        if self.training:
            '''要新加的loss加在这里'''

            # occ_lable_h = occ_label[:,0]
            # where = torch.not_equal(occ_lable_h, 1)
            # index_occ = torch.where(where)
            # h_num = torch.sum(1-occ_lable_h)
            # occ_feat = torch.zeros((int(feat_map_cls.size(0)/2),int(feat_map_cls.size(1)),int(h_num.item()),7)).to('cuda')
            # occ_feat[:,:,:,:] = feat_map_cls[int(feat_map_cls.size(0)/2):,:, index_occ[0].min().item():index_occ[0].max().item()+1 ,:]
            # occ_feat = self.occ_pooling(occ_feat).squeeze()
            occ_feat = None

        # 这里分离了occ_pre的梯度！
        feat_map = feat_map_cls.detach()
        x = self.base.avgpool(x.transpose(1, 2))  # B C 1
        global_feat = torch.flatten(x, 1)  ## B C
        h_feat = self.hen(feat_map).squeeze(dim=-1)  # B, C, 7


        h_local_0 = h_feat[..., 0]
        h_local_1 = h_feat[..., 1]
        h_local_2 = h_feat[..., 2]
        h_local_3 = h_feat[..., 3]
        h_local_4 = h_feat[..., 4]
        h_local_5 = h_feat[..., 5]
        h_local_6 = h_feat[..., 6]
        h_local_7 = h_feat[..., 7]

        occ_pre_label = []
        occ_pre_label.append(self.occ_predict1(h_local_0))
        occ_pre_label.append(self.occ_predict2(h_local_1))
        occ_pre_label.append(self.occ_predict3(h_local_2))
        occ_pre_label.append(self.occ_predict4(h_local_3))
        occ_pre_label.append(self.occ_predict5(h_local_4))
        occ_pre_label.append(self.occ_predict6(h_local_5))
        occ_pre_label.append(self.occ_predict7(h_local_6))
        occ_pre_label.append(self.occ_predict8(h_local_7))
        occ_pre_label = torch.cat(occ_pre_label, dim=0)

        '''用标签0,1做'''
        co_h = torch.softmax(occ_pre_label, dim=1)
        co_h_pro = co_h[:,1].view(8,-1)
        co_h = co_h.argmax(dim=1)  # (7B)
        # # (7B,2)；7B是batch里每个part（B,2）cat起来的！所以是七个局部的label* B叠起来
        co_h = co_h.view(8, -1)

        h_feat_cls = self.hen(feat_map_cls).squeeze(dim=-1)
        h_local_cls_0 = h_feat_cls[..., 0]
        h_local_cls_1 = h_feat_cls[..., 1]
        h_local_cls_2 = h_feat_cls[..., 2]
        h_local_cls_3 = h_feat_cls[..., 3]
        h_local_cls_4 = h_feat_cls[..., 4]
        h_local_cls_5 = h_feat_cls[..., 5]
        h_local_cls_6 = h_feat_cls[..., 6]
        h_local_cls_7 = h_feat_cls[..., 7]
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:

            cls_score = self.classifier(feat)
            return cls_score, global_feat, occ_feat, occ_lable_gt_2d,occ_lable_gt, occ_pre_label,torch.cat(
                [h_local_cls_0 / 8, h_local_cls_1 / 8, h_local_cls_2 / 8, h_local_cls_3 / 8, h_local_cls_4 / 8, h_local_cls_5 / 8,
                 h_local_cls_6 / 8,h_local_cls_7 / 8]).reshape(8, -1, 1024), rand_index, from_copy_tensor_label
        else:
            return global_feat, torch.cat(
                [h_local_cls_0 / 8, h_local_cls_1 / 8, h_local_cls_2 / 8, h_local_cls_3 / 8, h_local_cls_4 / 8, h_local_cls_5 / 8,
                 h_local_cls_6 / 8, h_local_cls_7 / 8]).reshape(8, -1, 1024), co_h, co_h_pro

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    elif cfg.MODEL.NAME == 'vit_oafr':
        if cfg.INPUT.SIZE_TRAIN == [224, 224]:
            model = build_vit_oafr_224(num_class, camera_num, view_num, cfg,__factory_T_type)
        else:
            model = build_vit_oafr_256(num_class, camera_num, view_num, cfg, __factory_T_type)
            # for key, value in model.named_parameters():
            #     print(key)
            # print('yes')
    elif cfg.MODEL.NAME == 'swin':
        model = build_swin(num_class, camera_num, view_num, cfg,__factory_T_type)
    elif cfg.MODEL.NAME == 'swin_oafr':
        if cfg.INPUT.SIZE_TRAIN == [224, 224]:
            model = build_swin_oafr(num_class, camera_num, view_num, cfg, __factory_T_type)
        else:
            model = build_swin_oafr_256(num_class, camera_num, view_num, cfg, __factory_T_type)
    elif cfg.MODEL.NAME == 'resnet50':
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')

    return model
