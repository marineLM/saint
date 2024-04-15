import torch
from torch import nn, Tensor
import numpy as np
from .pretrainmodel import SAINT


class MySAINT(nn.Module):
    """Note: In Grinsztajn et al (and TabSurvey) as well as the SAINT paper,
    AdamW is used while we use Adam here."""
    """
    Arguments
    ---------
    n_cont_features: int
        The number of continuous features.

    dim: int
        Size of the feature embeddings.

    depth: int
        Depth of the (Transformer) model. Number of stages.

    heads: int
        Number of attention heads in each Attention layer.

    attn_dropout: float
        The dropout level in attention layers of the Transformer.

    ff_dropout: float
        The dropout level in feed forward layers of the Transformer.

    attentiontype: str
        Variant of SAINT. 'col' refers to SAINT-s variant, 'row' is SAINT-i,
        and 'colrow' refers to SAINT.

    y_dim: int
            Should be 1 in the case of regression, and equal to the number of
            categories of the target otherwise
    """
    def __init__(self,
                 n_cont_features,
                 dim,
                 depth,
                 heads,
                 attn_dropout,
                 ff_dropout,
                 attentiontype='col',
                 y_dim=1
                 ):
        super().__init__()
        self.model = SAINT(
            categories=tuple([1]),  # no categorical data for now, only CLS
            num_continuous=n_cont_features,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=16,
            # dim_out=None,
            # mlp_hidden_mults=None,
            # mlp_act=None,
            num_special_tokens=0,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype=attentiontype,
            final_mlp_style='common',
            y_dim=y_dim
            )

    def forward(self, x: Tensor, n_categ: int = None) -> Tensor:
        # For now we do not allow fo categorical data.
        # So the only categorical data we can have is the CLS token.

        # Categorical data should be of type int to be able to use an Embedding
        # layer on it. Moreover, a tensor cannot have data of different dtype.
        # So we need to create a tensor of type int for the categorical data,
        # and a tensor of type float for the continuous data.
        x_cat = x[:, 0].reshape(-1, 1)
        x_cat = x_cat.to(torch.int)
        x_cont = x[:, 1:]

        # Impute by zeros and store the mask.
        # It should not matter what we use for imputation here as we will
        # replace these imputations by learned embeddings.
        con_mask = (~torch.isnan(x_cont)).to(torch.int)
        x_cont = torch.nan_to_num(x_cont)

        # Convert continuous data to embeddings.
        n_samples, n_cont_features = x_cont.shape
        x_cont_enc = torch.empty(n_samples, n_cont_features, self.model.dim)
        for i in range(self.model.num_continuous):
            x_cont_enc[:, i, :] = self.model.simple_MLP[i](x_cont[:, i])

        # Replace missing values in continuous data by their embeddings
        con_mask_temp = con_mask + self.model.con_mask_offset.type_as(con_mask)
        con_mask_temp = self.model.mask_embeds_cont(con_mask_temp)
        x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

        # Convert categorical data to embeddings.
        x_categ_enc = self.model.embeds(x_cat)

        reps = self.model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and apply
        # mlp on it in the next step to get the predictions.
        y_reps = reps[:, 0, :]
        y_outs = self.model.mlpfory(y_reps)

        # The output in case of regression should be (batch_size, ) rather than
        # (batch_size, 1) so we squeeze. Be careful whether this is ok for the
        # classification case.
        return y_outs.squeeze()
