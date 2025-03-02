import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import math

from  import *  # 确保你已经定义了  和相关的类或方法

__all__ = [
    'deit_small_Our_patch16_224', 
    'deit_small_Our__RandWeight_patch16_224'
]

class Our(VisionTransformer):
    def __init__(self, depth=12, num_heads=6, reduction=4, pool="avg", 
                 embed_dim=384, img_size=224, patch_size=16, AdaptiveAttentionFusion=None, 
                 feat_reduction=None, *args, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, *args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        img_size = to_2tuple(img_size)  # 确保这里使用的是正确的img_size
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        _params = dict(channel=depth*num_heads, reduction=reduction)
        if feat_reduction is not None:
            _params["feat_reduction"] = feat_reduction      
            _params["feats_channel"] = embed_dim//num_heads       
            _params["pool"] = pool
            
        self.adaptive_attention_fusion = AdaptiveAttentionFusion(**_params)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed_pat.shape[1]
        class_pos_embed = self.pos_embed_cls
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        # 适应不同尺寸输入的位置编码插值
        if w != h or npatch != N:  
            w0, h0 = w // self.patch_embed.patch_size[0], h // self.patch_embed.patch_size[0]
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            ).permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls
        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        attn_feats = []

        for i, blk in enumerate(self.blocks):
            x, weights_i, feat = blk(x)
            attn_feats.append(feat)
            attn_weights.append(weights_i)

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, attn_feats

    def forward(self, x, return_att=False, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, attn_feats = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        coarse_cam_pred = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)
        attn_feats = torch.stack(attn_feats)

        attn_weights_detach = attn_weights.detach().clone()
        k, b, h, n, m = attn_weights_detach.shape
        attn_weights_detach = attn_weights_detach.permute([1, 2, 0, 3, 4]).contiguous()
        attn_weights_detach = attn_weights_detach.view(b, h * k, n, m)

        attn_feats_detach = attn_feats.detach().clone()
        k, b, n, c = attn_feats_detach.shape
        attn_feats_detach = attn_feats_detach.view(k, b, n, -1, h)
        attn_feats_detach = attn_feats_detach.permute([1, 4, 0, 2, 3]).contiguous()
        attn_feats_detach = attn_feats_detach.view(b, h * k, n, -1)
        cross_attn_map, patch_attn_map = self.adaptive_attention_fusion(attn_feats_detach, attn_weights_detach)

        coarse_cam = x_patch.detach().clone()
        coarse_cam = F.relu(coarse_cam)

        n, c, h, w = coarse_cam.shape

        cross_attn = cross_attn_map.mean(1)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = cross_attn * coarse_cam
        elif attention_type == 'patchcam':
            cams = coarse_cam
        else:
            cams = cross_attn

        patch_attn = patch_attn_map.mean(1)[:, self.num_classes:, self.num_classes:]

        fine_cam = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0], cams.shape[1], -1, 1)).reshape(cams.shape[0], cams.shape[1], h, w)

        fine_cam_pred = self.avgpool(fine_cam).squeeze(3).squeeze(2)

        patch_attn = patch_attn.unsqueeze(0)

        cls_token_pred = x_cls.mean(-1)

        if return_att:
            return cls_token_pred, cams, patch_attn
        else:
            return cls_token_pred, coarse_cam_pred, fine_cam_pred

# 注册模型函数
@register_model
def deit_small_Our_patch16_224(pretrained=False, **kwargs):
    model = Our(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), AdaptiveAttentionFusion=, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model

@register_model
def deit_small_Our__RandWeight_patch16_224(pretrained=False, **kwargs):
    model = Our(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), AdaptiveAttentionFusion=_RandWeight, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model