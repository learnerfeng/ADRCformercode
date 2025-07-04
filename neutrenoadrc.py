import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from statistics import mean
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import _init_vit_weights, _load_weights
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
import copy


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., layerth=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.alpha = 0.6
        self.layerth = layerth

        # LESO parameters
        self.kpp = 4  # P控制器增益
        self.b0 = 2  # b0 设置为较大的值
        self.h1 = 0.1  # 离散时间步长
        self.omega0 = 10
        self.beta1 = 2 * self.omega0
        self.beta2 = self.omega0 *self.omega0

        # LESO states
        self.z1 = torch.tensor(0.0, requires_grad=False)  # 初始化为张量
        self.z2 = torch.tensor(0.0, requires_grad=False)
    def update_leso(self, v, u):
        """
        离散化 LESO 更新（基于误差计算的动态更新）
        :param v: 实际值（当前值）
        :param u: 控制信号
        """
        # 确保 self.z1 和 v 在同一设备
        device = v.device
        self.z1 = self.z1.to(device)
        self.z2 = self.z2.to(device)
        # 确保 v 是张量
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, requires_grad=False)

        e2 = (self.z1.detach() - v.detach()).mean()  # 误差 e2
        dz1 = self.z2 - self.beta1 * e2 + self.b0 * u
        dz2 = -self.beta2 * e2

        # 更新 LESO 状态
        self.z1 += dz1 * self.h1
        self.z2 += dz2 * self.h1
        print("run")
    def lsef(self, v0):
        """
        LSEF 计算公式
        :param v0: 目标值
        """
        error = (v0.detach() - self.z1.detach()).mean()  # 误差 error：目标值 - z1
        u0 = self.kpp * error  # 比例控制部分
        u = (u0 - self.z2) / self.b0  # 最终控制量
        return u

    def forward(self, x, v0=None):
        """
        前向传播，结合注意力机制和 LESO/LSEF 补偿
        :param x: 输入张量 (B, N, C)
        :param v0: 目标值 (B, N, C) 或者用于计算误差的参考值
        :return: 输出张量
        """
        B, N, C = x.shape
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 将 Q, K, V 分解

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 更新 LESO 状态和计算 LSEF 补偿
        if self.layerth > 0 and v0 is not None:
            v_mean = v.mean()  # 实际值（均值）
            self.update_leso(v=v_mean, u=0)  # 更新 LESO 状态
            res = self.lsef(v0=v0.mean())  # 计算 LSEF 补偿
        else:
            res = 0.0  # 如果不需要补偿，res 设置为 0

        # 最终结果：注意力加补偿
        x = (attn @ v) + res
        x = x.transpose(1, 2).reshape(B, N, C)

        # 投影层和丢弃
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.layerth == 0:
            return x, v  # 如果 layerth 为 0，返回 x 和 v
        else:
            return x  # 否则仅返回 x


class Block(nn.Module):
 
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, layerth= layerth)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layerth = layerth
 
    def forward(self, x, v0 = None):

        if self.layerth == 0:
            x_, v0 = self.attn(self.norm1(x))
        else:
            x_ = self.attn(self.norm1(x), v0 = v0)

        x = x + self.drop_path(x_)
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.layerth == 0:
            return x, v0
        else:
            return x
 
 
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
 
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
       
 
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim) ##### FIX IF YOU WANNA USE IMAGENET
        num_patches = self.patch_embed.num_patches
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, layerth = i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
 
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        self.init_weights(weight_init)
 
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)
 
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)
 
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
 
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
 
    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist
 
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x, v0 = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            x = self.blocks[i](x, v0 = v0)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
 
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
 
 



