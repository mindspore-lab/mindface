"""
vit
"""
from typing import Optional
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .helper import DropPath, trunc_normal_

__all__ = ['vit_t', 'vit_s', 'vit_b', 'vit_l']


class RandomMask(nn.Cell):
    """
    RandomMask.
    """
    def __init__(self, seed = 2022):
        super().__init__()
        self.sort = ops.Sort(axis = -1)
        self.rand = ops.UniformReal(seed = seed)
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.GatherD()
        self.tile = ops.Tile()
        self.ones = ops.Ones()
        self.cast = ops.Cast()
        self.mul = ops.Mul()

    def construct(self, x, mask_ratio=0.1):
        """
        construct.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = self.rand((N, L))

        ids_shuffle = self.sort(noise)[1]
        ids_restore = self.sort(self.cast(ids_shuffle, ms.float32))[1]

        mask = self.ones((N, L, D), ms.float32)
        mask[:, len_keep:, :] = 0
        ids_restore = self.expand_dims(ids_restore, -1)
        ids_restore = self.tile(ids_restore, (1,1,D))
        mask = self.gather(mask, 1, ids_restore)

        x_masked = self.mul(mask, x)

        return x_masked, mask

class Mlp(nn.Cell):
    """
    Mlp.
    """
    def __init__(self,
                in_features,
                hidden_features=None,
                out_features=None,
                act_layer=nn.ReLU6,
                drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(keep_prob = 1.0 - drop)

    def construct(self, x):
        """
        construct.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    """
    Attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(keep_prob = 1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob = 1 - proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """
        construct.
        """
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (batch_size, num_token, 3, self.num_heads,
                            embed_dim // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        k, q, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.attn_matmul_v(attn, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (batch_size, num_token, embed_dim))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Cell):
    """
    PatchEmbed.
    """
    def __init__(self, img_size=112, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                stride=patch_size, pad_mode='pad', has_bias=True)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        # self.cast = ops.Cast()

    def construct(self, x):
        """
        construct.
        """
        x = self.proj(x)
        batch_size, channels, height, width = x.shape
        x = self.reshape(x, (batch_size, channels, height * width))
        x = self.transpose(x, (0,2,1))
        return x


class ResidualCell(nn.Cell):
    """
    Cell which implements Residual function:

    $$output = x + f(x)$$

    Args:
        cell (Cell): Cell needed to add residual block.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = ResidualCell(nn.Dense(3,4))
    """

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x):
        """
        construct.
        """
        return self.cell(x) + x


class Block(nn.Cell):
    """
    Block.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_seed: int = 2022,
                 mlp_seed: int = 2022,
                 act_layer: nn.Cell = nn.ReLU6,
                 norm_layer: str = "ln"):
        super().__init__()

        if norm_layer == "bn":
            norm1 = nn.BatchNorm1d(num_features=num_patches)
            norm2 = nn.BatchNorm1d(num_features=num_patches)
        elif norm_layer == "ln":
            norm1 = nn.LayerNorm((dim,))
            norm2 = nn.LayerNorm((dim,))

        assert drop_path >= 0.

        self.layer1 = nn.SequentialCell([
            ResidualCell(
                nn.SequentialCell([
                    norm1,
                    Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop),
                    DropPath(keep_prob = 1 - drop_path, seed = attn_seed)])
                )
        ])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.layer2 = nn.SequentialCell([
            ResidualCell(
                nn.SequentialCell([
                    norm2,
                    Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop),
                    DropPath(keep_prob = 1 - drop_path, seed = mlp_seed)])
            )
        ])

    def construct(self, x):
        """
        construct.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class VisionTransformer(nn.Cell):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.

    Args:
        img_size (Int): The size of input images. Default: 112.
        patch_size (Int): The size of the patch. Default: 16.
        in_channels(Int): The number of the input channel. Default: 3.
        num_classes(Int): The number of classes. Default: 1000.
        embed_dim(Int): The dimension of embedded block. Default: 768.
        depth(Int): The depth of network. Default: 12.
        num_heads(Int): The number of the attention heads. Default: 12.
        mlp_ratio(Flaot): The ratio of mlp layers. Default: 4.
        qkv_bias(Bool): Bias of q, k and v. Default: False.
        qk_scale(Object): The scale of q and k. Default: None.
        drop_rate(Float): The rate of dropout. Default: 0.
        attn_drop_rate(Float): The rate of dropout in attention block. Default: 0.
        drop_path_rate(Float): The rate of path dropout. Default: 0.
        norm_layer(String): The type of norm layer. Default: "ln".
        mask_ratio(Float): The ratio of mask. Default: 0.1.

    Examples:
        >>> net = VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0, norm_layer="ln", mask_ratio=0.05)
    """
    def __init__(self,
                 img_size: int = 112,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer: str = "ln",
                 mask_ratio:float = 0.1,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.zeros = ops.Zeros()
        self.mul = ops.Mul()
        self.reshape = ops.Reshape()

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_channels=in_channels,
                                      embed_dim=embed_dim)

        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = ms.Parameter(
                trunc_normal_(
                    self.zeros((1, self.num_patches, embed_dim), ms.float32),
                    std=1.0),
                requires_grad=True)

        self.pos_drop = nn.Dropout(keep_prob = 1 - drop_rate)

        dpr = [i.item() for i in np.linspace(0, drop_path_rate, depth)]
        attn_seeds = [np.random.randint(2022) for _ in range(depth)]
        mlp_seeds = [np.random.randint(2022) for _ in range(depth)]

        blocks = []
        for i in range(depth):
            blocks.append(
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                      attn_seed = attn_seeds[i], mlp_seed = mlp_seeds[i],
                      norm_layer=norm_layer, num_patches=self.num_patches)
            )
        self.blocks = nn.SequentialCell(blocks)

        if norm_layer == "ln":
            self.norm = nn.LayerNorm((embed_dim,))
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(self.num_patches)

        self.feature = nn.SequentialCell(
            nn.Dense(in_channels=embed_dim * self.num_patches,
                     out_channels=embed_dim, has_bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Dense(in_channels=embed_dim, out_channels=num_classes, has_bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )

    def forward_features(self, x):
        """
        forward_features.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return self.reshape(x, (B, self.num_patches * self.embed_dim))

    def construct(self, x):
        """
        construct.
        """
        x = self.forward_features(x)
        x = self.feature(x)
        return x

def vit_t(num_features=512):
    """
    vit_t.

    Examples:
        >>> net = vit_t()
    """
    return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)


def vit_s(num_features=512):
    """
    vit_s.

    Examples:
        >>> net = vit_s()
    """
    return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)


def vit_b(num_features=512):
    """
    vit_b.

    Examples:
        >>> net = vit_b()
    """
    return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0, norm_layer="ln", mask_ratio=0.05)


def vit_l(num_features=512):
    """
    vit_l.

    Examples:
        >>> net = vit_l()
    """
    return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0, norm_layer="ln", mask_ratio=0.05)
