import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from matplotlib import pyplot as plt
import os
import numpy as np
import timm.models.vision_transformer


class PartialMAPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        # Inherit the parent class (Conv2d)
        super(PartialMAPConv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride,
                                            padding=padding, dilation=dilation,
                                            groups=groups, bias=bias,
                                            padding_mode=padding_mode)

        self.conv_beta = nn.Conv2d(in_channels, out_channels,
                                kernel_size+2, stride=stride,
                                padding=padding+1, dilation=dilation,
                                groups=groups, bias=bias,
                                padding_mode=padding_mode)

        self.conv_gamma = nn.Conv2d(in_channels, out_channels,
                                kernel_size-2, stride=stride,
                                padding=padding-1, dilation=dilation,
                                groups=groups, bias=bias,
                                padding_mode=padding_mode)

        # .conv2d(conved, self.weight, self.bias, self.stride,
        #                   self.padding, self.dilation, self.groups)

        # Define the kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                              * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None
        # Initialize the weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        with torch.no_grad():
            if self.mask_kernel.type() != img.type():
                self.mask_kernel = self.mask_kernel.to(img)
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))
            self.update_mask = F.conv2d(mask, self.mask_kernel,
                                        bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)
            # calcurate mask ratio (sum(1) / sum(M))
            self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # calcurate WT . (X * M)
        conved = torch.mul(img, mask)
        beta = self.conv_beta(conved)
        gamma = self.conv_gamma(conved)
        conved = F.conv2d(conved, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        # print("self.stride: {}, self.padding: {}, self.dilation: {}, self.groups: {}".format(self.stride, self.padding, self.dilation, self.groups))
        # print("conved.shape: ", conved.shape)
        # print("beta.shape: ", beta.shape)
        # print("gamma.shape: ", gamma.shape)
        conved = conved / 3 + beta / 3 + gamma / 3
        if self.bias is not None:
            # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
            # The masked part pixel is updated to 0
            output = torch.mul(output, self.mask_ratio)
        else:
            # Multiply WT . (X * M) and sum(1) / sum(M)
            output = torch.mul(conved, self.mask_ratio)

        return output, self.update_mask


class UpsampleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the upsampling layer with nearest neighbor
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_for_6 = nn.Upsample(scale_factor=1.75, mode='nearest')

    def forward(self, layer_num, dec_feature, enc_feature, dec_mask, enc_mask):
        # upsample and concat features
        # print("Layer Num: ", layer_num)
        if layer_num == 6:
            out = self.upsample_for_6(dec_feature)
            out_mask = self.upsample_for_6(dec_mask)
        else:
            out = self.upsample(dec_feature)
            out_mask = self.upsample(dec_mask)
        # print("In Upsample] dec_feature:{} | out: {} | enc_feature: {}".format(dec_feature.shape, out.shape, enc_feature.shape))
        out = torch.cat([out, enc_feature], dim=1)
        # upsample and concat masks
        out_mask = torch.cat([out_mask, enc_mask], dim=1)
        return out, out_mask


class MAPConvActiv(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', dec=False,
                 bn=True, active='relu', conv_bias=False):
        super().__init__()
        # Define the partial conv layer
        if sample == 'down-7':
            params = {"kernel_size": 7, "stride": 2, "padding": 3}
        elif sample == 'down-5':
            params = {"kernel_size": 5, "stride": 2, "padding": 2}
        elif sample == 'down-3':
            params = {"kernel_size": 3, "stride": 2, "padding": 1}
        else:
            params = {"kernel_size": 3, "stride": 1, "padding": 1}
        self.conv = PartialMAPConv2d(in_ch, out_ch,
                                  params["kernel_size"],
                                  params["stride"],
                                  params["padding"],
                                  bias=conv_bias)

        # Define other layers
        if dec:
            self.upcat = UpsampleConcat()
        if bn:
            bn = nn.BatchNorm2d(out_ch)
        if active == 'relu':
            self.activation = nn.ReLU()
        elif active == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, layer_num, img, mask, enc_img=None, enc_mask=None):
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(layer_num, img, enc_img, mask, enc_mask)
            out, update_mask = self.conv(out, update_mask)
        else:
            out, update_mask = self.conv(img, mask)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out, update_mask


class MAPConvUNet(nn.Module):
    def __init__(self, finetune, in_ch=3, layer_size=7):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False
        self.layer_size = layer_size

        self.enc_1 = MAPConvActiv(in_ch, 64, 'down-7', bn=False)
        self.enc_2 = MAPConvActiv(64, 128, 'down-5')
        self.enc_3 = MAPConvActiv(128, 256, 'down-5')
        self.enc_4 = MAPConvActiv(256, 512, 'down-3')
        self.enc_5 = MAPConvActiv(512, 512, 'down-3')
        self.enc_6 = MAPConvActiv(512, 512, 'down-3')
        self.enc_7 = MAPConvActiv(512, 512, 'down-3')
        # self.enc_8 = MAPConvActiv(512, 512, 'down-3')

        # self.dec_8 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_7 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_6 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_5 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_4 = MAPConvActiv(512+256, 256, dec=True, active='leaky')
        self.dec_3 = MAPConvActiv(256+128, 128, dec=True, active='leaky')
        self.dec_2 = MAPConvActiv(128+64,   64, dec=True, active='leaky')
        self.dec_1 = MAPConvActiv(64+3,      3, dec=True, bn=False,
                                active=None, conv_bias=True)

    def forward(self, img, mask):
        enc_f, enc_m = [img], [mask]
        # print("Before MAPConvUNet Encoder Image is NaN? : {} | mask is Nan? {}".format(torch.isnan(img).any(), torch.isnan(mask).any()))
        # print("----------------------------------------------------------------------------")
        for layer_num in range(1, self.layer_size+1):
            if layer_num == 1:
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(layer_num, img, mask)
            else:
                enc_f.append(feature)
                enc_m.append(update_mask)
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(layer_num, feature,
                                                              update_mask)
            feature += 1e-8
        #     print("In MAPConvUNet Encoder [{}] featrue is NaN? : {} | mask is Nan? {}".format(layer_num, torch.isnan(feature).any(), torch.isnan(update_mask).any()))

        # print("----------------------------------------------------------------------------")

        assert len(enc_f) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size+1)):
            feature, update_mask = getattr(self, 'dec_{}'.format(layer_num))(
                    layer_num, feature, update_mask, enc_f.pop(), enc_m.pop())
            feature += 1e-8
        #     print("In MAPConvUNet Decoder [{}] featrue is NaN? : {} | mask is Nan? {}".format(layer_num, torch.isnan(feature).any(), torch.isnan(update_mask).any()))

        # print("*"*60)
        return feature, mask

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        if not self.freeze_enc_bn:
            return 
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()

class MAPConvUNet_TwoPhase(nn.Module):
    def __init__(self, finetune, in_ch=3, layer_size=6):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False
        self.layer_size = layer_size

        self.enc_phase1_1 = MAPConvActiv(in_ch, 64, 'down-7', bn=False)
        self.enc_phase1_2 = MAPConvActiv(64, 128, 'down-5')
        self.enc_phase1_3 = MAPConvActiv(128, 256, 'down-5')
        self.enc_phase1_4 = MAPConvActiv(256, 512, 'down-3')
        self.enc_phase1_5 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase1_6 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase1_7 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase1_8 = MAPConvActiv(512, 512, 'down-3')

        self.dec_phase1_8 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase1_7 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase1_6 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase1_5 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase1_4 = MAPConvActiv(512+256, 256, dec=True, active='leaky')
        self.dec_phase1_3 = MAPConvActiv(256+128, 128, dec=True, active='leaky')
        self.dec_phase1_2 = MAPConvActiv(128+64,   64, dec=True, active='leaky')
        self.dec_phase1_1 = MAPConvActiv(64+3,      3, dec=True, bn=False,
                                active=None, conv_bias=True)

        self.enc_phase2_1 = MAPConvActiv(in_ch, 64, 'down-7', bn=False)
        self.enc_phase2_2 = MAPConvActiv(64, 128, 'down-5')
        self.enc_phase2_3 = MAPConvActiv(128, 256, 'down-5')
        self.enc_phase2_4 = MAPConvActiv(256, 512, 'down-3')
        self.enc_phase2_5 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase2_6 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase2_7 = MAPConvActiv(512, 512, 'down-3')
        self.enc_phase2_8 = MAPConvActiv(512, 512, 'down-3')

        self.dec_phase2_8 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase2_7 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase2_6 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase2_5 = MAPConvActiv(512+512, 512, dec=True, active='leaky')
        self.dec_phase2_4 = MAPConvActiv(512+256, 256, dec=True, active='leaky')
        self.dec_phase2_3 = MAPConvActiv(256+128, 128, dec=True, active='leaky')
        self.dec_phase2_2 = MAPConvActiv(128+64,   64, dec=True, active='leaky')
        self.dec_phase2_1 = MAPConvActiv(64+3,      3, dec=True, bn=False,
                                active=None, conv_bias=True)


    def forward(self, img, mask):
        enc_f_Phase1, enc_m_Phase1 = [img], [mask]
        enc_f_Phase2, enc_m_Phase2 = [img], [mask]

        ## phase 1
        for layer_num in range(1, self.layer_size+1):
            if layer_num == 1:
                feature_phase1, update_mask_phase1 = \
                    getattr(self, 'enc_phase1_{}'.format(layer_num))(layer_num, img, mask)
            else:
                enc_f_Phase1.append(feature_phase1)
                enc_m_Phase1.append(update_mask_phase1)
                feature_phase1, update_mask_phase1 = \
                    getattr(self, 'enc_phase1_{}'.format(layer_num))(layer_num, feature_phase1,
                                                              update_mask_phase1)

        assert len(enc_f_Phase1) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size+1)):
            feature_phase1, update_mask_phase1 = getattr(self, 'dec_phase1_{}'.format(layer_num))(
                    layer_num, feature_phase1, update_mask_phase1, enc_f_Phase1.pop(), enc_m_Phase1.pop())


        ## phase 2
        for layer_num in range(1, self.layer_size+1):
            if layer_num == 1:
                feature_phase2, update_mask_phase2 = \
                    getattr(self, 'enc_phase2_{}'.format(layer_num))(layer_num, feature_phase1, mask)
            else:
                enc_f_Phase2.append(feature_phase2)
                enc_m_Phase2.append(update_mask_phase2)
                feature_phase2, update_mask_phase2 = \
                    getattr(self, 'enc_phase2_{}'.format(layer_num))(layer_num, feature_phase2,
                                                              update_mask_phase2)

        assert len(enc_f_Phase2) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size+1)):
            feature_phase2, update_mask_phase2 = getattr(self, 'dec_phase2_{}'.format(layer_num))(
                    layer_num, feature_phase2, update_mask_phase2, enc_f_Phase2.pop(), enc_m_Phase2.pop())

        return feature_phase1, feature_phase2, mask

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        if not self.freeze_enc_bn:
            return 
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()


class VisionFaceTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionFaceTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        self.MAPConvUNet = MAPConvUNet(finetune=False)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # print("[In patchify] patch size: ", self.patch_embed.patch_size)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_features(self, x):
        raw_img = x.clone()
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        represented_feature = x[:, 1:]

        represented_feature = self.unpatchify(represented_feature)
        ####
        return represented_feature

    def forward(self, x, mask):
        feature = self.forward_features(x)
        # print("forward_features: ", feature.shape, feature.dtype, torch.isnan(feature).any())

        feature, _ = self.MAPConvUNet(feature, mask)
        # print("MAPConvUNet: ", feature.shape, feature.dtype, torch.isnan(feature).any())

        return feature, mask

# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


if __name__ == '__main__':
    # from utils import init_xavier
    size = (1, 3, 224, 224)
    img = torch.ones(size)
    mask = torch.ones(size)
    mask[:, :, 64:-64, :][:, :, :, 64:-64] = 0

    conv = PartialMAPConv2d(3, 3, 7, 2, 3)
    criterion = nn.L1Loss()
    img.requires_grad = True

    output, out_mask = conv(img, mask)


    # loss = criterion(output, torch.randn(size))
    # loss.backward()

    # print(img.grad[0])
    # assert (torch.sum(torch.isnan(conv.weight.grad)).item() == 0)
    # assert (torch.sum(torch.isnan(conv.bias.grad)).item() == 0)

    # model = MAPConvUNet(False, layer_size=7)
    model = VisionFaceTransformer()
    # before = model.enc_5.conv.weight[0][0]
    # print("before: ", before)
    # model.apply(init_xavier)
    # after = model.enc_5.conv.weight[0][0]
    # print(after - before)
    output = model(img, mask, img)
    print(output.shape)
