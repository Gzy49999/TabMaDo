import torch
import torch.nn as nn

from .embedding_tree import NeuralEmbeddingTree
from .plr_layer import PeriodicEmbeddings

class EmbeddingLayer(nn.Module):
    def __init__(self, num,dimension, config,device = "cuda:0"):
        """Embedding layer that handles numerical and categorical embeddings.

        Parameters
        ----------
        feature_info : dict
            Dictionary where keys are feature names and values are their respective input dimensions.
        config : Config
            Configuration object containing all required settings.
        """
        super().__init__()
        self.device = device
        self.d_model = getattr(config, "d_model", 128)
        self.embedding_activation = getattr(
            config, "embedding_activation", nn.Identity()
        )
        self.layer_norm_after_embedding = getattr(
            config, "layer_norm_after_embedding", False
        )
        self.embedding_projection = getattr(config, "embedding_projection", True)
        self.use_cls = getattr(config, "use_cls", False)
        self.cls_position = getattr(config, "cls_position", 0)
        self.embedding_dropout = (
            nn.Dropout(getattr(config, "embedding_dropout", 0.0))
            if getattr(config, "embedding_dropout", None) is not None
            else None
        )
        self.embedding_type = getattr(config, "embedding_type", "plr")
        self.embedding_bias = getattr(config, "embedding_bias", False)

        # Sequence length
        self.seq_len = num
        self.dimension = dimension

        # Initialize numerical embeddings based on embedding_type
        if self.embedding_type == "ndt":
            self.embeddings = nn.ModuleList(
                [
                    NeuralEmbeddingTree(self.dimension, self.d_model)
                    for _ in range(num)
                ]
            )
        elif self.embedding_type == "plr":
            self.embeddings = PeriodicEmbeddings(
                n_features=self.seq_len,
                d_embedding=self.d_model,
                n_frequencies=getattr(config, "n_frequencies", 48),
                frequency_init_scale=getattr(config, "frequency_init_scale", 0.01),
                activation=True,
                lite=getattr(config, "plr_lite", False),
            )
        elif self.embedding_type == "linear":
            self.embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.dimension,
                            self.d_model,
                            bias=self.embedding_bias,
                        ),
                        self.embedding_activation,
                    )
                    for _ in range(num)
                ]
            )
        # for splines and other embeddings
        # splines followed by linear if n_knots actual knots is less than the defined knots
        else:
            raise ValueError(
                "Invalid embedding_type. Choose from 'linear', 'ndt', or 'plr'."
            )

        self.embeddings.to(self.device)
        # Class token if required
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Layer normalization if required
        if self.layer_norm_after_embedding:
            self.embedding_norm = nn.LayerNorm(self.d_model)


    def forward(self,data):
        """Defines the forward pass of the model.

        Parameters
        ----------
        data: tuple of lists of tensors

        Returns
        -------
        Tensor
            The output embeddings of the model.

        Raises
        ------
        ValueError
            If no features are provided to the model.
        """
        embeddings = [ ]
        # Class token initialization
        if self.use_cls:
            batch_size = (
                data[0].size(0)  # type: ignore
            )  # type: ignore
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.embedding_type == "plr":
            if data is not None:
                embeddings = self.embeddings(data)
                if self.layer_norm_after_embedding:
                    embeddings = self.embedding_norm(embeddings)

        else:
            if self.embeddings and data is not None:
                for i, emb in enumerate(self.embeddings):
                    embeddings_i = emb(data[:,i])
                    embeddings.append(embeddings_i)
                embeddings = torch.stack(embeddings, dim=1)
                if self.layer_norm_after_embedding:
                    embeddings = self.embedding_norm(embeddings)
        if isinstance(embeddings, torch.Tensor):
            embeddings = [embeddings]  # 将单个张量包装为列表
        x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]

        # Add class token if required
        if self.use_cls:
            if self.cls_position == 0:
                x = torch.cat([cls_tokens, x], dim=1)  # type: ignore
            elif self.cls_position == 1:
                x = torch.cat([x, cls_tokens], dim=1)  # type: ignore
            else:
                raise ValueError(
                    "Invalid cls_position value. It should be either 0 or 1."
                )

        # Apply dropout to embeddings if specified in config
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        return x




