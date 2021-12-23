""" Full assembly of the parts to form the complete network """

from __future__ import division

import os

import torch
import torch.nn as nn

from modeling.model_utils.da_att import DANetHead
from modeling.model_utils.bifusion import BiFusion,Attention_block
from modeling.model_utils.unet_parts import *
from modeling.model_utils.transformer import *
from modeling.model_utils import vit_seg_configs as seg_configs
from utils.utils import init_weights,count_param

# class TransUNet(nn.Module):
#     def __init__(self, backbone,config, BatchNorm, output_stride, num_classes,img_size, freeze_bn=False):
#         super(TransUNet, self).__init__()
#
#         self.backbone = backbone
#         self.n_classes = num_classes
#         self.conv1 = nn.Conv2d(2048, 512, 1, bias=False)
#         self.output_stride = output_stride
#
#         self.avepool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         # self.transformer = Transformer(img_size//16, patch_size=2, in_channels=1024, out_channels=1024,
#         #                                num_heads=16,hidden_size=512, num_layers=10, vis=False)
#         self.transformer = Transformer(config, img_size, vis=False)
#
#         self.up1 = Up(1024, 512 //2, BatchNorm)
#         self.up2 = Up(512, 256 // 4, BatchNorm)
#         self.up3 = Up(128, 64, BatchNorm)
#         self.outc = OutConv(64, num_classes)
#
#         if freeze_bn:
#             self.freeze_bn()
#     def forward(self, x):
#         x2 = x[5]#n*64*256*256
#         x3 = x[3]#n*256*128*128
#         x4 = x[2]#n*512*64*64
#         x4_5 = x[1]  # n*1024*32*32
#         x5 = self.conv1(x[0])
#         x_tran = self.transformer(x5)#n*512*2*2
#         x = self.up0(x_tran, x4_5)  # n*512*32*32
#         x = self.up1(x, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         logits = self.outc(x)
#         x = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
#
#         return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                # logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

def _transUnet(backbone, BatchNorm, output_stride, num_classes,img_size):
    n_skip=3
    vit_name='R50-ViT-B_16'
    vit_patches_size= 16
    config_vit = CONFIGS[vit_name]
    config_vit.n_skip = n_skip
    config_vit.n_classes = num_classes
    # snapshot_path='./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    model = VisionTransformer(config_vit, img_size, num_classes)
    model.load_from(weights=np.load(config_vit.pretrained_path))
    return model

def transUnet(backbone, BatchNorm, output_stride, num_classes,img_size, freeze_bn=False):
    return _transUnet(backbone, BatchNorm, output_stride, num_classes,img_size)

CONFIGS = {
    'ViT-B_16': seg_configs.get_b16_config(),
    'ViT-B_32': seg_configs.get_b32_config(),
    'ViT-L_16': seg_configs.get_l16_config(),
    'ViT-L_32': seg_configs.get_l32_config(),
    'ViT-H_14': seg_configs.get_h14_config(),
    'R50-ViT-B_16': seg_configs.get_r50_b16_config(),
    'R50-ViT-L_16': seg_configs.get_r50_l16_config(),
    'testing': seg_configs.get_testing(),
}
