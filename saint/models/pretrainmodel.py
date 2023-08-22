from .model import *


class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class SAINT(nn.Module):
    """Arguments
    ---------
    categories: tuple of size the number of categorical columns.
        Gives the number of categories for each categorical column.

    num_continuous: int
        The number of continuous columns.

    dim: int
        Size of the feature embeddings.

    depth: int
        Depth of the (Transformer) model. Number of stages.

    heads: int
        Number of attention heads in each Attention layer.

    dim_head: int
        Used by `Transformer`.

    dim_out: int
        Used as the dimension of the output of self.mlp, but I do not see where
        self.mlp is used.

    mlp_hidden_mults: tuple of int
        Integers by which to multiply the input to obtain the size of hidden
        layers. Used for the dimension of self.mlp but again, I do not see
        where this MLP is used.

    mlp_act: an activation layer
        Again used for sel.mlp

    num_special_tokens: int

    attn_dropout: float
        The dropout level in attention layers of the Transformer.

    ff_dropout: float
        The dropout level in feed forward layers of the Transformer.

    cont_embeddings: str
        Style of embedding continuous data. The continuous data is actually
        embedded using augmentations/embed_data_mask.py. Only the value 'MLP'
        of this parameter will work. It transforms each feature value (a float)
        into a `dim` dimensional vector using a MLP (a different one for each
        feature) with one hidden layer of 100 hidden units (as hard-coded in
        the SAINT architecture).

    scalingfactor: unused

    attentiontype: str
        Variant of SAINT. 'col' refers to SAINT-s variant, 'row' is SAINT-i,
        and 'colrow' refers to SAINT.

    final_mlp_style: str
        Choice of MLP after the Transformer. If `common`, .... Else, ...
        (I do not understand yet what the final MLP does.)

    y_dim: int
        Should be 1 in the case of regression, and equal to the number of
        categories of the target otherwise.
    """
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=0,
        attn_dropout=0.,
        ff_dropout=0.,
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col',
        final_mlp_style='common',
        y_dim=2
        ):

        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct
        # position in the categories embedding table
        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        # The first line pads categories on the left with one element of value
        # `num_special_tokens`
        # for ex, with categories = (1, 2, 5, 6) and num_special_tokens=0
        # categories_offset = tensor([0, 1, 2, 5, 6])
        # categories_offset = tensor([0, 1, 3, 8])

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList(
                [simple_MLP([1, 100, self.dim])
                 for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList(
                [simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(
            self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP(
                [dim, (self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous,
                                np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP([
            dim*(self.num_continuous+self.num_categories),
            6*dim*(self.num_continuous+self.num_categories)//5,
            dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([
            dim*(self.num_continuous+self.num_categories),
            6*dim*(self.num_continuous+self.num_categories)//5,
            dim*(self.num_continuous+self.num_categories)//2])

    def forward(self, x_categ, x_cont):
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:, :self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories:, :])
        return cat_outs, con_outs
