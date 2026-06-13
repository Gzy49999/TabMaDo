"""
Code was adapted from https://github.com/Yura52/rtdl
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from .Mambular_utils.embedding_layer import EmbeddingLayer
from .Mambular_utils.mamba import Mamba
from .Mambular_utils.mlp_utils import MLPhead
from .Mambular_utils.config import TabulambaConfig
from .Mambular_utils.basemodel import BaseModel
import numpy as np
import typing as ty
import torch.nn.init as nn_init

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )

def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )

class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.
        This variation of MLP was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers
        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, d_in, d_layers, num_classes, dropout, dim_t = 128, t_in=False):
        super().__init__()
        self.t_in = t_in
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=dim_t)

        if num_classes > 2:
            self.d_out = num_classes
        else:
            self.d_out = 1

        self.proj = nn.Linear(d_in, dim_t)

        if self.t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, dim_t)
            )

        self.head = nn.Linear(dim_t, self.d_out)

    # def forward(self, x, t=None):
    #     if self.t_in and t is not None:
    #         emb = self.time_embed(timestep_embedding(t, self.dim_t))
    #         x = self.proj(x) + emb
    #     else:
    #         x = self.proj(x)
    #     x = self.mlp(x)
    #     x = self.head(x)
    #     if self.num_classes > 2:
    #         return torch.softmax(x, dim=1)
    #     else:
    #         return torch.sigmoid(x)

    def forward(self, x, t=None, return_embedding=False):
        if self.t_in and t is not None:
            emb_t = self.time_embed(timestep_embedding(t, self.dim_t))
            x = self.proj(x) + emb_t
        else:
            x = self.proj(x)

        embedding = self.mlp(x)  # (batch_size, dim_t)
        x = self.head(embedding)
        if self.num_classes > 2:
            logits = torch.softmax(x, dim=1)
        else:
            logits = torch.sigmoid(x)

        if return_embedding:
            return logits, embedding
        return logits


class MLPClassifierWithPrototype(nn.Module):
    """带原型损失的MLP分类器"""

    def __init__(self, d_in, d_layers, num_classes, dropout, dim_t=128, t_in=True, proto_weight=0.1):
        super().__init__()
        self.t_in = t_in
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.proto_weight = proto_weight

        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=dim_t)

        if num_classes > 2:
            self.d_out = num_classes
        else:
            self.d_out = 1

        self.proj = nn.Linear(d_in, dim_t)

        self.prototype_proj = nn.Linear(d_in, dim_t)

        if self.t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, dim_t)
            )

        self.head = nn.Linear(dim_t, self.d_out)

    def forward(self, x, t=None, prototype=None, y=None):
        if self.t_in and t is not None:
            emb_t = self.time_embed(timestep_embedding(t, self.dim_t))
            x = self.proj(x) + emb_t
        else:
            x = self.proj(x)

        if prototype is not None:
            proto_feat = self.prototype_proj(prototype)
            x = x + proto_feat

        embedding = self.mlp(x)

        logits = self.head(embedding)
        if self.num_classes > 2:
            logits = torch.softmax(logits, dim=1)
        else:
            logits = torch.sigmoid(logits)

        proto_loss = torch.tensor(0.0, device=x.device)
        if prototype is not None and y is not None:
            proto_feat_for_loss = self.prototype_proj(prototype)

            embedding_norm = F.normalize(embedding, dim=1)
            proto_norm = F.normalize(proto_feat_for_loss, dim=1)

            cosine_sim = (embedding_norm * proto_norm).sum(dim=1)

            minority_mask = (y == 1)
            majority_mask = (y == 0)

            if minority_mask.sum() > 0:
                proto_loss = proto_loss + (1 - cosine_sim[minority_mask]).mean()

            if majority_mask.sum() > 0:
                target_sim = 0.0
                proto_loss = proto_loss + F.relu(cosine_sim[majority_mask] - target_sim).mean()

        return logits, proto_loss

class MLPEncoder(nn.Module):
    def __init__(self, d_in, d_layers, d_out, dropout, dim_t = 128, t_in = False):
        super().__init__()
        self.dim_t = dim_t
        self.t_in = t_in
        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_out)

        self.proj = nn.Linear(d_in, dim_t)
        if t_in:
            self.time_embed = nn.Sequential(
                nn.Linear(dim_t, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, dim_t)
            )
        #self.head = nn.Linear(dim_t, self.d_out)

    def forward(self, x, t=None):
        x = self.proj(x)
        if self.t_in and t is not None:
            emb = self.time_embed(timestep_embedding(t, self.dim_t))
            x = x + emb
        x = self.mlp(x)
        return x

'''----------for FT-Transformer---------'''
def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )

def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )

class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x

class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x

'''-------------end-----------'''


class ClassifierWithCondScorer(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, cond_dim, feature_dim=128,
                 dropout=0.0, margin=0.3, pos_weight=3.0):
        super().__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.margin = margin
        self.pos_weight = pos_weight

        self.encoder = MLPEncoder(
            d_in=d_in,
            d_layers=d_hidden,
            d_out=feature_dim,
            dropout=dropout,
            dim_t=feature_dim,
            t_in=True
        )

        self.head = nn.Linear(feature_dim, 1 if n_classes == 2 else n_classes)

        self.prototype_encoder = MLPEncoder(
            d_in=cond_dim,
            d_layers=d_hidden,
            d_out=feature_dim,
            dropout=dropout,
            dim_t=feature_dim,
            t_in=False
        )

        self.temperature = nn.Parameter(torch.ones([]) * 0.1)

    def forward(self, x_t, t, cond_anchor=None, y=None):
        losses = {}

        x_feature = self.encoder(x_t, t)  # (B, D)

        logits = self.head(x_feature)
        if self.n_classes == 2:
            logits = logits.squeeze()
            pos_weight_tensor = torch.tensor([self.pos_weight]).to(x_t.device)
            losses['cls_loss'] = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight_tensor
            )
        else:
            losses['cls_loss'] = F.cross_entropy(logits, y.long())

        if cond_anchor is not None and y is not None:
            prototype_feature = self.prototype_encoder(cond_anchor, t=None)
            prototype = prototype_feature[0:1].detach()  # (1, D)

            x_norm = F.normalize(x_feature, dim=1)
            p_norm = F.normalize(prototype, dim=1)

            similarity = (x_norm * p_norm).sum(dim=1)  # (B,)

            minority_mask = (y == 1)
            majority_mask = (y == 0)

            loss_margin = 0.0
            if minority_mask.sum() > 0:
                pos_sim = similarity[minority_mask]
                loss_margin += F.relu(self.margin - pos_sim).mean()

            if majority_mask.sum() > 0:
                neg_sim = similarity[majority_mask]
                loss_margin += F.relu(neg_sim - self.margin).mean()

            if minority_mask.sum() > 0 and majority_mask.sum() > 0:
                sim_gap = similarity[minority_mask].mean() - similarity[majority_mask].mean()
                loss_margin = loss_margin + F.relu(0.1 - sim_gap)

            losses['contrast_loss'] = loss_margin
        else:
            losses['contrast_loss'] = torch.tensor(0.0, device=x_t.device)

        return logits, losses

    def get_guide_gradients(self, x_t, t, cond_anchor, target_cls=0, guide_strength=0.5):
        """
        生成阶段：计算双引导梯度（分类器+CondScorer）
        Returns:
            total_grad: 融合后的引导梯度 (B, d_in)
        """
        x_in = x_t.detach()
        x_in.requires_grad_(True)

        x_feature = self.encoder(x_in, t)
        logits = self.head(x_feature)
        if self.n_classes == 2:
            logits = logits.squeeze()
            prob = torch.sigmoid(logits)

            if target_cls == 1:
                cls_loss = -prob.sum()
            else:
                cls_loss = prob.sum()
        else:
            cls_loss = -logits[:, target_cls].sum()
        cls_loss.backward(retain_graph=True)
        cls_grad = x_in.grad.detach().clone()
        x_in.grad.zero_()

        model_device = next(self.prototype_encoder.parameters()).device
        cond_anchor = cond_anchor.to(model_device)
        cond_feature = self.prototype_encoder(cond_anchor, t=None)
        # x_feature_norm = x_feature / x_feature.norm(dim=1, keepdim=True)
        # cond_feature_norm = cond_feature / cond_feature.norm(dim=1, keepdim=True)
        # sim_loss = -torch.diag(cond_feature_norm @ x_feature_norm.t()).mean()
        x_norm = F.normalize(x_feature, dim=1)
        cond_norm = F.normalize(cond_feature, dim=1)

        if target_cls == 1:
            sim_loss = -(x_norm * cond_norm).sum(dim=1).mean()
        else:
            sim_loss = (x_norm * cond_norm).sum(dim=1).mean()
        sim_loss.backward()
        sim_grad = x_in.grad.detach().clone()
        x_in.grad.zero_()
        cls_grad = cls_grad / (cls_grad.norm(dim=1, keepdim=True) + 1e-8)
        sim_grad = sim_grad / (sim_grad.norm(dim=1, keepdim=True) + 1e-8)
        total_grad = (cls_grad + sim_grad) / 2

        total_grad = guide_strength * total_grad / (total_grad.norm(dim=1, keepdim=True) + 1e-8)
        x_t.requires_grad_(False)
        print(f"t={t[0].item():3d} | cls_grad_norm={cls_grad.norm().item():.4f} | "
              f"sim_grad_norm={sim_grad.norm().item():.4f} | "
              f"total_grad_norm={total_grad.norm().item():.4f}")
        return total_grad


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, d_layers, dropout, dim_t=128):  # num_classes=0, is_y_cond=False):
        super().__init__()
        self.dim_t = dim_t

        # d0 = rtdl_params['d_layers'][0]

        self.mlp = MLP.make_baseline(d_in=dim_t, d_layers=d_layers, dropout=dropout, d_out=d_in)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb
        return self.mlp(x)


class FTTransfMLPDiffusion(nn.Module):
    def __init__(self, d_in, d_layers, dropout, dim_t=128):
        super().__init__()
        self.dim_t = dim_t

        # 准备RTDL参数
        rtdl_params = {
            'd_layers': d_layers,
            'dropout': dropout,
        }
        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        # transformer参数
        n_layers = 3
        d_token = 128
        n_heads = 8
        d_ffn_factor = 0.5
        attention_dropout = dropout
        ffn_dropout = dropout
        residual_dropout = 0.0
        activation = "reglu"
        prenormalization = True
        initialization = "kaiming"

        # tokenizer
        self.tokenizer = Tokenizer(d_in, None, d_token, True)  # categories=None
        n_tokens = self.tokenizer.n_tokens

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_in)

        # MLP for final output
        self.mlp = MLP.make_baseline(**rtdl_params)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x, timesteps):
        """
        Args:
            x: (batch_size, d_in) - input features
            timesteps: (batch_size,) - diffusion timesteps
        Returns:
            (batch_size, d_in) - output
        """
        x_cat = None
        x = self.tokenizer(x, x_cat)  # (batch_size, n_tokens, d_token)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                None,  # key_compression
                None,  # value_compression
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]  # (batch_size, d_token)

        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)  # (batch_size, d_in)

        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb  # (batch_size, dim_t)

        return self.mlp(x)

class MambularDiffusion(BaseModel):
    def __init__(
        self,
        d_in,
        dimension,
        dim_t =64,
        config: TabulambaConfig = TabulambaConfig(),
        num_classes=2,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False
        self.d_in = d_in
        self.dimension = dimension
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.embedding_layer = EmbeddingLayer(
            d_in,
            dimension,
            config=config,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.mamba = Mamba(config, num_classes=num_classes,)
        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=self.d_in,
        )
        if self.hparams.shuffle_embeddings:
            self.perm = torch.randperm(self.embedding_layer.seq_len)

        self.initialize_pooling_layers(config=config, n_inputs=self.d_in)

    def forward(self, x, timesteps, class_labels=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        t_emb = emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = self.embedding_layer(x) + t_emb
        if self.hparams.shuffle_embeddings:
            x = x[:, self.perm, :]
        x = self.mamba(x, class_labels=class_labels)

        x = self.pool_sequence(x)

        preds = self.tabular_head(x)

        return preds
