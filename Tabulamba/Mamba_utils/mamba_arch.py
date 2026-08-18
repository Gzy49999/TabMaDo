import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Mamba_utils.get_norm_fn import get_normalization_layer
from ..Mamba_utils.normalization_layers import LayerNorm, LearnableLayerScaling, RMSNorm

# Heavily inspired and mostly taken from https://github.com/alxndrTL/mamba.py

#通过堆叠多个残差块来处理输入数据
class Mamba(nn.Module):
    """Mamba model composed of multiple MambaBlocks.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of MambaBlocks constituting the model.
    """

    def __init__(
        self,
        config,
        num_classes=2,
        device = "cuda:0"
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model=getattr(config, "d_model", 128),
                    expand_factor=getattr(config, "expand_factor", 4),
                    bias=getattr(config, "bias", True),
                    d_conv=getattr(config, "d_conv", 4),
                    conv_bias=getattr(config, "conv_bias", False),
                    dropout=getattr(config, "dropout", 0.0),
                    dt_rank=getattr(config, "dt_rank", "auto"),
                    d_state=getattr(config, "d_state", 256),
                    dt_scale=getattr(config, "dt_scale", 1.0),
                    dt_init=getattr(config, "dt_init", "random"),
                    dt_max=getattr(config, "dt_max", 0.1),
                    dt_min=getattr(config, "dt_min", 1e-04),
                    dt_init_floor=getattr(config, "dt_init_floor", 1e-04),
                    norm=get_normalization_layer(config),  # type: ignore
                    activation=getattr(config, "activation", nn.SiLU()),
                    bidirectional=getattr(config, "bidirectional", False),
                    use_learnable_interaction=getattr(
                        config, "use_learnable_interaction", True
                    ),
                    layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
                    AD_weight_decay=getattr(config, "AD_weight_decay", True),
                    BC_layer_norm=getattr(config, "BC_layer_norm", False),
                    use_pscan=getattr(config, "use_pscan", False),
                    dilation=getattr(config, "dilation", 1),
                    num_classes=num_classes,
                )
                for _ in range(getattr(config, "n_layers", 6))
            ]
        )

    def forward(self, x, class_labels=None):
        self.layers.to(self.device)
        x.to(self.device)
        for layer in self.layers:
            x = layer(x, class_labels=class_labels)

        return x

#一个残差块，由一个 MambaBlock 和一个归一化层组成
class ResidualBlock(nn.Module):
    """Residual block composed of a MambaBlock and a normalization layer.

    Parameters
    ----------
    d_model : int, optional
        Dimension of the model input, by default 32.
    expand_factor : int, optional
        Expansion factor for the model, by default 2.
    bias : bool, optional
        Whether to use bias in the MambaBlock, by default False.
    d_conv : int, optional
        Dimension of the convolution layer in the MambaBlock, by default 16.
    conv_bias : bool, optional
        Whether to use bias in the convolution layer, by default True.
    dropout : float, optional
        Dropout rate for the layers, by default 0.01.
    dt_rank : Union[str, int], optional
        Rank for dynamic time components, 'auto' or an integer, by default 'auto'.
    d_state : int, optional
        Dimension of the state vector, by default 32.
    dt_scale : float, optional
        Scale factor for dynamic time components, by default 1.0.
    dt_init : str, optional
        Initialization strategy for dynamic time components, by default 'random'.
    dt_max : float, optional
        Maximum value for dynamic time components, by default 0.1.
    dt_min : float, optional
        Minimum value for dynamic time components, by default 1e-03.
    dt_init_floor : float, optional
        Floor value for initialization of dynamic time components, by default 1e-04.
    norm : callable, optional
        Normalization layer, by default RMSNorm.
    activation : callable, optional
        Activation function used in the MambaBlock, by default `F.silu`.
    bidirectional : bool, optional
        Whether the block is bidirectional, by default False.
    use_learnable_interaction : bool, optional
        Whether to use learnable interactions, by default False.
    layer_norm_eps : float, optional
        Epsilon for layer normalization, by default 1e-05.
    AD_weight_decay : bool, optional
        Whether to apply weight decay in adaptive dynamics, by default False.
    BC_layer_norm : bool, optional
        Whether to use layer normalization for batch compatibility, by default False.
    use_pscan : bool, optional
        Whether to use PSCAN, by default False.

    Attributes
    ----------
    layers : MambaBlock
        The main MambaBlock layers for processing input.
    norm : callable
        Normalization layer applied before the MambaBlock.

    Methods
    -------
    forward(x)
        Performs a forward pass through the block and returns the output.

    Raises
    ------
    ValueError
        If the provided normalization layer is not valid.
    """

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=False,
        use_pscan=False,
        dilation=1,
        num_classes=2,
        **kwargs
    ):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
        }

        # Check if the provided normalization layer is valid
        if isinstance(norm, type) and norm.__name__ not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm.__name__}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )
        elif isinstance(norm, str) and norm not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)

        self.layers = MambaBlock(
            d_model=d_model,
            expand_factor=expand_factor,
            bias=bias,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            dt_rank=dt_rank,  # type: ignore
            d_state=d_state,
            dt_scale=dt_scale,
            dt_init=dt_init,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_init_floor=dt_init_floor,
            activation=activation,
            bidirectional=bidirectional,
            use_learnable_interaction=use_learnable_interaction,
            layer_norm_eps=layer_norm_eps,
            AD_weight_decay=AD_weight_decay,
            BC_layer_norm=BC_layer_norm,
            use_pscan=use_pscan,
            dilation=dilation,
            num_classes=num_classes,  # 传递2
            minority_class=1,  # 少数类是1
        )
        self.norm = norm

    def forward(self, x, class_labels=None):
        """Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the block.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual connection and MambaBlock.
        """
        output = self.layers(self.norm(x), class_labels=class_labels)
        return output + x


class MambaBlock(nn.Module):
    """MambaBlock module containing the main computational components for processing input.

    Parameters
    ----------
    d_model : int, optional
        Dimension of the model input, by default 32.
    expand_factor : int, optional
        Factor by which the input is expanded in the block, by default 2.
    bias : bool, optional
        Whether to use bias in the linear projections, by default False.
    d_conv : int, optional
        Dimension of the convolution layer, by default 16.
    conv_bias : bool, optional
        Whether to use bias in the convolution layer, by default True.
    dropout : float, optional
        Dropout rate applied to the layers, by default 0.01.
    dt_rank : Union[str, int], optional
        Rank for dynamic time components, either 'auto' or an integer, by default 'auto'.
    d_state : int, optional
        Dimensionality of the state vector, by default 32.
    dt_scale : float, optional
        Scale factor applied to the dynamic time component, by default 1.0.
    dt_init : str, optional
        Initialization strategy for the dynamic time component, by default 'random'.
    dt_max : float, optional
        Maximum value for dynamic time component initialization, by default 0.1.
    dt_min : float, optional
        Minimum value for dynamic time component initialization, by default 1e-03.
    dt_init_floor : float, optional
        Floor value for dynamic time component initialization, by default 1e-04.
    activation : callable, optional
        Activation function applied in the block, by default `F.silu`.
    bidirectional : bool, optional
        Whether the block is bidirectional, by default False.
    use_learnable_interaction : bool, optional
        Whether to use learnable feature interaction, by default False.
    layer_norm_eps : float, optional
        Epsilon for layer normalization, by default 1e-05.
    AD_weight_decay : bool, optional
        Whether to apply weight decay in adaptive dynamics, by default False.
    BC_layer_norm : bool, optional
        Whether to use layer normalization for batch compatibility, by default False.
    use_pscan : bool, optional
        Whether to use the PSCAN mechanism, by default False.

    Attributes
    ----------
    in_proj : nn.Linear
        Linear projection applied to the input tensor.
    conv1d : nn.Conv1d
        1D convolutional layer for processing input.
    x_proj : nn.Linear
        Linear projection applied to input-dependent tensors.
    dt_proj : nn.Linear
        Linear projection for the dynamical time component.
    A_log : nn.Parameter
        Logarithmically stored tensor A for internal dynamics.
    D : nn.Parameter
        Tensor for the D component of the model's dynamics.
    out_proj : nn.Linear
        Linear projection applied to the output.
    learnable_interaction : LearnableFeatureInteraction
        Layer for learnable feature interactions, if `use_learnable_interaction` is True.

    Methods
    -------
    forward(x)
        Performs a forward pass through the MambaBlock.
    """

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,#是否在线性层中使用偏置
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",#动态时间组件的秩
        d_state=32,
        dt_scale=1.0,#动态时间组件的缩放因子，用于调整初始化的标准差
        dt_init="random",#动态时间组件的初始化策略
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        activation=F.silu,
        bidirectional=False,#是否为双向模型
        use_learnable_interaction=False,#是否使用可学习的特征交互
        layer_norm_eps=1e-05,#层归一化的 epsilon 值
        AD_weight_decay=False,#是否在动态时间组件中使用权重衰减
        BC_layer_norm=False,#是否使用批兼容的层归一化
        use_pscan=False,#是否使用 PSCAN 机制
        dilation=1,#卷积层的膨胀率
        num_classes=2,  # 改为2
        minority_class=1,  # 少数类的标签值
        majority_state_ratio=0.2,  # 多数类状态占比
        enable_state_gain=True,  # 方案1开关
        enable_state_allocation=False,  # 方案2开关
        enable_reset_gate=False,  # 方案3开关
        enable_bi_weight=True,  # 方案4开关
    ):
        super().__init__()

        self.use_pscan = use_pscan

        if self.use_pscan:
            try:
                from mambapy.pscan import pscan  # type: ignore

                self.pscan = pscan  # Store the imported pscan function
            except ImportError:
                self.pscan = None  # Set to None if pscan is not available
                print(
                    "The 'mambapy' package is not installed. Please install it by running:\n"
                    "pip install mambapy"
                )
        else:
            self.pscan = None

        self.d_inner = d_model * expand_factor
        self.bidirectional = bidirectional
        self.use_learnable_interaction = use_learnable_interaction

        #前向输入投影层，将输入数据从维度 d_model 映射到 2 * d_inner
        self.in_proj_fwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        #如果模型是双向的，还会有一个反向输入投影层
        if self.bidirectional:
            self.in_proj_bwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        #前向一维卷积层，用于处理输入数据，对输入数据的每个通道独立地应用卷积核，从而提取局部特征
        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,#对每个通道独立地应用卷积核
            padding=d_conv - 1,#在输入数据的两侧各填充 d_conv - 1 个零
        )
        #如果模型是双向的，还会有一个反向卷积层
        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.d_inner,
                padding=d_conv - 1,
                dilation=dilation,
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if self.use_learnable_interaction:
            self.learnable_interaction = LearnableFeatureInteraction(self.d_inner)

        #将输入数据从维度 self.d_inner 投影到维度 dt_rank + 2 * d_state
        self.x_proj_fwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)  # type: ignore
        if self.bidirectional:
            self.x_proj_bwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)  # type: ignore
        #动态时间组件的投影层，用于处理动态时间信息
        self.dt_proj_fwd = nn.Linear(dt_rank, self.d_inner, bias=True)  # type: ignore
        if self.bidirectional:
            self.dt_proj_bwd = nn.Linear(dt_rank, self.d_inner, bias=True)  # type: ignore

        #根据 dt_init 参数的值，使用不同的初始化策略初始化动态时间组件的权重
        #逆平方根缩放（dt_rank**-0.5）
        dt_init_std = dt_rank**-0.5 * dt_scale # type: ignore
        #将权重初始化为一个常数值
        #将 self.dt_proj_fwd.weight 和 self.dt_proj_bwd.weight 的所有元素初始化为 dt_init_std
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_fwd.weight, dt_init_std)
            if self.bidirectional:
                nn.init.constant_(self.dt_proj_bwd.weight, dt_init_std)
        #将权重初始化为均匀分布的随机值
        #所有元素初始化为从范围 [-dt_init_std, dt_init_std] 中均匀采样的随机值
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_fwd.weight, -dt_init_std, dt_init_std)
            if self.bidirectional:
                nn.init.uniform_(self.dt_proj_bwd.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        self.num_classes = num_classes  # 2
        self.minority_class = minority_class  # 1
        self.enable_state_gain = enable_state_gain
        self.enable_state_allocation = enable_state_allocation
        self.enable_reset_gate = enable_reset_gate
        self.enable_bi_weight = enable_bi_weight

        # ========== 方案1：状态更新增益（二分类）==========
        if enable_state_gain:
            # 形状: [2, d_state]
            self.state_update_gain = nn.Parameter(torch.ones(num_classes, d_state))
            with torch.no_grad():
                # 多数类(0)增益=1.0，少数类(1)增益=2.0（更高）
                self.state_update_gain[minority_class] = 2.0

        # ========== 方案2：类别特定状态分配（二分类）==========
        if enable_state_allocation and num_classes == 2 and d_state >= 4:
            d_state_total = d_state
            majority_d_state = int(d_state_total * majority_state_ratio)
            minority_d_state = d_state_total - majority_d_state

            self.class_d_state = {
                0: majority_d_state,  # 多数类用少量状态
                1: minority_d_state,  # 少数类用更多状态
            }

            # 为每个类别创建独立的A和D参数
            self.class_A_log = nn.ParameterDict({
                str(c): nn.Parameter(torch.log(
                    torch.arange(1, self.class_d_state[c] + 1, dtype=torch.float32).repeat(self.d_inner, 1)
                ))
                for c in range(num_classes)
            })
            self.class_D = nn.ParameterDict({
                str(c): nn.Parameter(torch.ones(self.d_inner))
                for c in range(num_classes)
            })

        # ========== 方案3：重置门控（二分类）==========
        if enable_reset_gate:
            self.reset_gate = nn.Linear(d_model, 1)
            self.reset_threshold = 0.5

        # ========== 方案4：双向融合权重（二分类）==========
        if enable_bi_weight and bidirectional:
            # 形状: [2]，多数类(0)更依赖前向，少数类(1)更依赖反向
            self.bidirectional_class_weight = nn.Parameter(torch.tensor([0.3, 0.7]))



        #初始化动态时间组件（dt_proj_fwd）的偏置项（bias），生成一个特定范围内的值，并将其赋值给偏置项
        # torch.rand(self.d_inner)：生成一个形状为[self.d_inner]的随机张量，其值在[0, 1) 范围内。
        # math.log(dt_max) - math.log(dt_min)：计算对数范围的宽度。
        # torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))：将随机值缩放到对数范围的宽度内，得到一个对数尺度上的随机值。
        # + math.log(dt_min)：将对数尺度上的随机值平移到[log({dt_min}),  log({dt_max})] 范围内。
        # torch.exp(...)：将对数尺度上的值转换回线性尺度，得到一个在[{dt_min}, {dt_max}] 范围内的随机值。
        # .clamp(min=dt_init_floor)：使用clamp函数确保生成的值不会低于dt_init_floor，这是一个下限值，用于防止过小的值
        dt_fwd = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # 计算逆时间步长 inv_dt_fwd
        #torch.expm1 是一个数值稳定的函数，用于计算 exp(x)−1
        #-torch.expm1(-dt_fwd)：计算 1 - exp(-{dt_fwd})
        inv_dt_fwd = dt_fwd + torch.log(-torch.expm1(-dt_fwd))
        #torch.no_grad()：一个上下文管理器，用于暂停梯度计算。这在初始化参数时非常有用，因为初始化过程不需要计算梯度
        with torch.no_grad():
            self.dt_proj_fwd.bias.copy_(inv_dt_fwd)

        if self.bidirectional:
            dt_bwd = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt_bwd = dt_bwd + torch.log(-torch.expm1(-dt_bwd))
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(inv_dt_bwd)

        #生成一个从 1 到 d_state 的一维张量，包含 d_state 个连续整数，数据类型为 float32
        #将上述一维张量重复 self.d_inner 次，形成一个二维张量
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        #对矩阵 A 的每个元素取自然对数，将结果转换为一个 PyTorch 参数（nn.Parameter），
        # 这意味着这个张量会被自动添加到模型的参数列表中，并在训练过程中参与梯度计算和更新
        self.A_log_fwd = nn.Parameter(torch.log(A))
        #生成一个形状为 [self.d_inner] 的一维张量，所有元素初始化为 1
        #将结果转换为一个 PyTorch 参数，同样会参与训练过程中的梯度计算和更新
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))

        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A))
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        if not AD_weight_decay:
            self.A_log_fwd._no_weight_decay = True  # type: ignore
            self.D_fwd._no_weight_decay = True  # type: ignore

        if self.bidirectional:
            if not AD_weight_decay:
                self.A_log_bwd._no_weight_decay = True  # type: ignore
                self.D_bwd._no_weight_decay = True  # type: ignore

        #输出投影层，将内部维度 d_inner 映射回原始输入维度 d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dt_rank = dt_rank
        self.d_state = d_state

        if BC_layer_norm:
            self.dt_layernorm = RMSNorm(self.dt_rank, eps=layer_norm_eps)  # type: ignore
            self.B_layernorm = RMSNorm(self.d_state, eps=layer_norm_eps)
            self.C_layernorm = RMSNorm(self.d_state, eps=layer_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        # ========== 带改进的SSM方法（二分类）==========

    def ssm_with_improvements(self, x, class_labels=None, forward=True):
        batch_size, L, _ = x.shape

        # 获取投影
        if forward:
            deltaBC = self.x_proj_fwd(x)
            delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj_fwd(delta))

            if self.enable_state_allocation and class_labels is not None:
                # 二分类：根据类别选择A和D
                if class_labels.dim() > 1:
                    class_labels = class_labels.squeeze()
                class_id = class_labels[0].item() if class_labels.numel() == 1 else class_labels[0].item()
                A = -torch.exp(self.class_A_log[str(class_id)].float())
                D = self.class_D[str(class_id)].float()
            else:
                A = -torch.exp(self.A_log_fwd.float())
                D = self.D_fwd.float()
        else:
            deltaBC = self.x_proj_bwd(x)
            delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj_bwd(delta))

            if self.enable_state_allocation and class_labels is not None:
                if class_labels.dim() > 1:
                    class_labels = class_labels.squeeze()
                class_id = class_labels[0].item() if class_labels.numel() == 1 else class_labels[0].item()
                A = -torch.exp(self.class_A_log[str(class_id)].float())
                D = self.class_D[str(class_id)].float()
            else:
                A = -torch.exp(self.A_log_bwd.float())
                D = self.D_bwd.float()

        # 选择性扫描计算
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))

        # 方案1：状态更新增益（二分类）
        if self.enable_state_gain and class_labels is not None:
            if class_labels.dim() > 1:
                class_labels = class_labels.squeeze()
            # 确保 class_labels 是 [batch_size] 形状
            if class_labels.dim() == 0:  # 标量情况
                class_labels = class_labels.unsqueeze(0).expand(batch_size)
            elif class_labels.shape != (batch_size,):
                class_labels = class_labels.view(batch_size)

            # 获取增益：多数类(0)=1.0，少数类(1)=2.0
            gain = self.state_update_gain[class_labels]  # [batch, d_state]
            gain = gain.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, d_state]
        else:
            gain = 1.0

        # 方案3：动态重置（二分类：多数类重置，少数类保持）
        if self.enable_reset_gate:
            reset_prob = torch.sigmoid(self.reset_gate(x))
            reset = (reset_prob > self.reset_threshold).float().unsqueeze(-1)
            if class_labels is not None:
                # 多数类(0)重置，少数类(1)不重置
                is_majority = (class_labels == 0).float().unsqueeze(-1).unsqueeze(-1)
                reset = reset * is_majority
        else:
            reset = torch.zeros(batch_size, L, 1, 1, device=x.device)

        # 循环扫描
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        hs = []

        for t in range(L):
            h = deltaA[:, t] * h * (1 - reset[:, t]) + BX[:, t] * gain
            hs.append(h)

        hs = torch.stack(hs, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x

        return y

    # ========== 修改 forward 方法 ==========
    def forward(self, x, class_labels=None):
        """修改后的 forward，支持类别标签（二分类）"""
        _, L, _ = x.shape

        # 前向投影
        xz_fwd = self.in_proj_fwd(x)
        x_fwd, z_fwd = xz_fwd.chunk(2, dim=-1)
        x_fwd = x_fwd.transpose(1, 2)
        x_fwd = self.conv1d_fwd(x_fwd)[:, :, :L]
        x_fwd = x_fwd.transpose(1, 2)

        if self.bidirectional:
            xz_bwd = self.in_proj_bwd(x)
            x_bwd, z_bwd = xz_bwd.chunk(2, dim=-1)
            x_bwd = x_bwd.transpose(1, 2)
            x_bwd = self.conv1d_bwd(x_bwd)[:, :, :L]
            x_bwd = x_bwd.transpose(1, 2)

        if self.use_learnable_interaction:
            x_fwd = self.learnable_interaction(x_fwd)
            if self.bidirectional:
                x_bwd = self.learnable_interaction(x_bwd)

        x_fwd = self.activation(x_fwd)
        x_fwd = self.dropout(x_fwd)

        y_fwd = self.ssm_with_improvements(x_fwd, class_labels=class_labels, forward=True)

        if self.bidirectional:
            x_bwd = self.activation(x_bwd)
            x_bwd = self.dropout(x_bwd)
            y_bwd = self.ssm_with_improvements(torch.flip(x_bwd, [1]), class_labels=class_labels, forward=False)
            y_bwd = torch.flip(y_bwd, [1])

            # 方案4：类别依赖的双向融合（二分类）
            if self.enable_bi_weight and class_labels is not None:
                if class_labels.dim() > 1:
                    class_labels = class_labels.squeeze()
                # 多数类(0)权重低(0.3)，少数类(1)权重高(0.7)
                weight = self.bidirectional_class_weight[class_labels]
                weight = weight.unsqueeze(1).unsqueeze(2)
                y = y_fwd * weight + y_bwd * (1 - weight)
            else:
                y = (y_fwd + y_bwd) / 2
        else:
            y = y_fwd

        # 输出门控
        z_fwd = self.activation(z_fwd)
        z_fwd = self.dropout(z_fwd)
        output = y * z_fwd
        output = self.out_proj(output)

        return output

    # def forward(self, x):
    #     #L：序列长度
    #     #x：输入张量，形状为 [batch_size, sequence_length, d_model]
    #     _, L, _ = x.shape
    #     #投影后的张量，形状为 [batch_size, sequence_length, 2 * d_inner]。
    #     xz_fwd = self.in_proj_fwd(x)
    #     #沿最后一个维度分成两个部分，形状为 [batch_size, sequence_length, d_inner]
    #     x_fwd, z_fwd = xz_fwd.chunk(2, dim=-1)
    #
    #     #将 x_fwd 的形状从 [batch_size, sequence_length, d_inner] 转换为 [batch_size, d_inner, sequence_length]，以适应 nn.Conv1d 的输入要求
    #     x_fwd = x_fwd.transpose(1, 2)
    #     #[:, :, :L]：截取卷积结果的前 L 个时间步，以保持序列长度不变
    #     x_fwd = self.conv1d_fwd(x_fwd)[:, :, :L]
    #     #再转换回原来的形状
    #     x_fwd = x_fwd.transpose(1, 2)
    #
    #     if self.bidirectional:
    #         xz_bwd = self.in_proj_bwd(x)
    #         x_bwd, z_bwd = xz_bwd.chunk(2, dim=-1)
    #
    #         x_bwd = x_bwd.transpose(1, 2)
    #         x_bwd = self.conv1d_bwd(x_bwd)[:, :, :L]
    #         x_bwd = x_bwd.transpose(1, 2)
    #
    #     if self.use_learnable_interaction:
    #         x_fwd = self.learnable_interaction(x_fwd)
    #         if self.bidirectional:
    #             x_bwd = self.learnable_interaction(x_bwd)  # type: ignore
    #
    #     #对 x_fwd 应用非线性变换
    #     x_fwd = self.activation(x_fwd)
    #     #Dropout 层，用于防止过拟合
    #     x_fwd = self.dropout(x_fwd)
    #     #forward=True：指示 SSM 进行前向传播
    #     y_fwd = self.ssm(x_fwd, forward=True)
    #
    #     if self.bidirectional:
    #         x_bwd = self.activation(x_bwd)  # type: ignore
    #         x_bwd = self.dropout(x_bwd)
    #         #torch.flip(x_bwd, [1])：将 x_bwd 沿时间维度翻转，以便 SSM 可以处理反向序列
    #         y_bwd = self.ssm(torch.flip(x_bwd, [1]), forward=False)
    #         #将前向和反向的结果相加
    #         y = y_fwd + torch.flip(y_bwd, [1])
    #         #对结果进行平均，以保持数值稳定
    #         y = y / 2
    #     else:
    #         y = y_fwd
    #
    #     z_fwd = self.activation(z_fwd)
    #     z_fwd = self.dropout(z_fwd)
    #
    #     output = y * z_fwd
    #     output = self.out_proj(output)
    #
    #     return output

    #对输入的张量 dt、B 和 C 应用层归一化
    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def ssm(self, x, forward=True):
        if forward:
            #A 和 D 是状态空间模型的参数，分别表示系统的动态矩阵和输出矩阵
            A = -torch.exp(self.A_log_fwd.float())
            D = self.D_fwd.float()
            deltaBC = self.x_proj_fwd(x)
            delta, B, C = torch.split(
                deltaBC,
                [self.dt_rank, self.d_state, self.d_state],  # type: ignore
                dim=-1,
            )
            delta, B, C = self._apply_layernorms(delta, B, C)
            #delta 通过 self.dt_proj_fwd 进行线性变换，然后应用 F.softplus 激活函数，确保 delta 的值为正
            delta = F.softplus(self.dt_proj_fwd(delta))
        # 反向传播
        else:
            A = -torch.exp(self.A_log_bwd.float())
            D = self.D_bwd.float()
            deltaBC = self.x_proj_bwd(x)
            delta, B, C = torch.split(
                deltaBC,
                [self.dt_rank, self.d_state, self.d_state],  # type: ignore
                dim=-1,
            )
            delta, B, C = self._apply_layernorms(delta, B, C)
            delta = F.softplus(self.dt_proj_bwd(delta))

        y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        #delta.unsqueeze(-1)：将 delta 的形状从 [batch_size, sequence_length, dt_rank] 变为 [batch_size, sequence_length, dt_rank, 1]
        #A：动态矩阵，形状为 [d_inner, d_state]
        #计算 delta 和 A 的逐元素乘积后取指数，形状为 [batch_size, sequence_length, d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        #将 B 的形状从 [batch_size, sequence_length, d_state] 变为 [batch_size, sequence_length, 1, d_state]
        #计算 delta 和 B 的逐元素乘积，形状为 [batch_size, sequence_length, dt_rank, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        #将 x 的形状从 [batch_size, sequence_length, d_model] 变为 [batch_size, sequence_length, d_model, 1]
        #BX形状为 [batch_size, sequence_length, dt_rank, d_state]
        BX = deltaB * (x.unsqueeze(-1))

        if self.use_pscan:
            hs = self.pscan(deltaA, BX)  # type: ignore
        else:
            #初始化隐藏状态 h 为零张量，形状为 [batch_size, d_inner, d_state]
            h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device)
            hs = []

            for t in range(0, L):
                h = deltaA[:, t] * h + BX[:, t]
                hs.append(h)

            hs = torch.stack(hs, dim=1)
        ##计算隐藏状态 hs 和输出矩阵 C 的矩阵乘积，形状为 [batch_size, sequence_length, d_inner, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

#学习特征之间的交互作用
class LearnableFeatureInteraction(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        #n_vars：输入特征的数量
        #interaction_weights：一个可学习的参数矩阵，形状为 (n_vars, n_vars)，用于表示特征之间的交互权重
        self.interaction_weights = nn.Parameter(torch.Tensor(n_vars, n_vars))
        #使用 Xavier 均匀初始化方法初始化 interaction_weights
        nn.init.xavier_uniform_(self.interaction_weights)

    def forward(self, x):
        batch_size, n_vars, d_model = x.size()
        interactions = torch.matmul(x, self.interaction_weights)
        return interactions.view(batch_size, n_vars, d_model)


