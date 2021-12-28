""" Full assembly of the parts to form the complete network """

from __future__ import division
import torch.nn.functional as F
import os
from modeling.model_utils.transformer import *
from modeling.model_utils import vit_seg_configs as seg_configs
from modeling.model_utils.aspp import build_aspp
from modeling.model_utils.decoder import build_decoder
from modeling.model_utils.da_att import DANetHead

class VisionTransformer(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)

        self.decoder1 = Decodercov(config)
        self.aspp = build_aspp(backbone, 512 , output_stride, BatchNorm)
        self.decoder2 = build_decoder(num_classes, 'resnet50', BatchNorm)
        in_channels = 512
        # self.net = multi_head_attention_2d(in_channels, in_channels, in_channels, num_classes, 4, 0.5, 'SAME')
        self.head = DANetHead(512, num_classes, BatchNorm)
        self.output_stride = output_stride
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x= self.decoder1(x)
        x = self.aspp(x)
        low_level_feat=features[1]
        x0=self.head(features[0])
        x = self.decoder2(x, low_level_feat)
        x0 = F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True)
        x = x0 + x
        x = F.interpolate(x, scale_factor=self.output_stride / 4, mode='bilinear', align_corners=True)
        return x
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

class Decodercov(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        return x

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

    model = VisionTransformer(backbone,BatchNorm, output_stride, config_vit, img_size, num_classes)
    model.load_from(weights=np.load(config_vit.pretrained_path))
    return model

def transdeep(backbone, BatchNorm, output_stride, num_classes,img_size, freeze_bn=False):
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
