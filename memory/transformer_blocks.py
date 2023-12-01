import torch
from einops import rearrange

class Alibi(torch.nn.Module):
    """
    Class for the positional embedding from the Alibi paper:

    https://arxiv.org/pdf/2108.12409.pdf

    Supports cross-attention!

    Parameters
    ----------
    context_x : int
        The context length in the x-direction
    context_y : int, optional
        The context length in the y-direction, by default None
    n_heads : int, optional
        The number of heads in the MHA, by default 8
    """
    def __init__(self,
                 context_x,
                 context_y = None,
                 n_heads = 8,):
        super().__init__()
        self.context_x = context_x
        if context_y is None:
            context_y = context_x
        self.context_y = context_y
        self.x_longer_context = context_x > context_y

        self.n_heads = n_heads

        # scaling based on paper TODO: maybe allow other values than 2 ** -8
        n_sequence = torch.arange(start = n_heads, end = 0, step = -1)
        self.head_scalars = 2 ** (-8 / n_sequence)

        self.M = self._create_M()

        self.requires_grad_(False)

    def _create_M(self):
        """
        Creates the positional embedding analog matrix from Alibi paper,
        with capacity for cross-attention / different x and y context lengths.

        I imagine there is a better torch-native way to do this,
        but it only runs once so I'm not concerned about the for-loops.
        """
        if self.x_longer_context:
            lower_len = self.context_y
            diff = self.context_x - self.context_y
            axis = 1
        else:
            lower_len = self.context_x
            diff = self.context_y - self.context_x
            axis = 0

        M = torch.zeros(lower_len, lower_len)
        for i in range(1, lower_len):
            M += torch.diag(-i * torch.ones(lower_len - i), -i)
        
        # symmetrize
        M += M.clone().T

        # append values matching the pattern for the longer context
        if diff > 0:
            for i in range(diff):
                vec = torch.arange(-lower_len - i, -i)
                M = torch.cat((M, vec.unsqueeze(axis)), axis = axis)
        
        M = M[None, :] * self.head_scalars[:, None, None]

        return M.permute(0, 2, 1)
    
    def get_M(self, crop = None):
        """
        Returns the positional embedding matrix.

        Parameters
        ----------
        crop : int | Tuple, optional
            The number of rows and columns to crop from the matrix, by default None
        """
        M = self.M
        if crop is not None:
            if isinstance(crop, int):
                crop = (crop, crop)
            M = M[:, :crop[0], :crop[1]]
        return M.unsqueeze(0)

class Attention(torch.nn.Module):
    """
    Standard multi-head attention module.

    Parameters
    ----------
    dim : int
        The dimension of the input and output
    dim_head : int, optional
        The dimension of the subspace for each head, by default 64
    n_heads : int, optional
        The number of heads, by default 8
    dropout : float, optional
        The dropout rate, by default 0.
    bias : bool, optional
        Whether to use bias in the linear layers, by default False"""
    def __init__(self, 
                 dim,
                 dim_head = 64,
                 n_heads = 8,
                 dropout = 0.,
                 bias = False,
                 context_x = 32,
                 context_y = None,
                 has_pos_emb = True,
                 alibi = True):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.n_heads = n_heads
        self.dropout = dropout
        self.inner_dim = dim_head * n_heads

        self.norm = torch.nn.LayerNorm(dim)

        self.W_q = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_k = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_v = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_o = torch.nn.Linear(self.inner_dim, dim, bias = bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.alibi = alibi
        self.alibi_obj = None
        self.cross_attention = False if context_y is None else True

        self.has_pos_emb = has_pos_emb

        if self.has_pos_emb:
            self._init_pos_emb(context_x, context_y)
        

    def _init_pos_emb(self, context_x, context_y):
        self.context = context_x
        if self.alibi:
            self.alibi_obj = Alibi(context_x, context_y, n_heads = self.n_heads)
        else:
            if context_y is None:
                self.pos_emb = torch.nn.Parameter(torch.randn(1, context_x, self.dim))
            else:
                self.pos_emb_x = torch.nn.Parameter(torch.randn(1, context_x, self.dim))
                self.pos_emb_y = torch.nn.Parameter(torch.randn(1, context_y, self.dim))

    #TODO : implement masking for autoregressive here
    def forward(self, x, y = None):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        # storing cause it's used twice
        add_pos_emb = self.has_pos_emb and not self.alibi

        # TODO: might need to flip x and y?
        if self.cross_attention:
            assert y is not None, "Cross attention requires two inputs"
            if add_pos_emb:
                x += self.pos_emb_x
                y += self.pos_emb_y
            q, k, v = self.W_q(x), self.W_k(y), self.W_v(y)
        else:
            if add_pos_emb:
                x += self.pos_emb
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

        attention = torch.einsum("b h i k, b h j k -> b h i j", q, k)
        attention = attention / (self.dim_head ** 0.5)

        if self.alibi:
            # sinple :)
            _, _, crop_x, crop_y = attention.shape
            attention += self.alibi_obj.get_M(crop = (crop_x, crop_y))

        attention = self.dropout(attention.softmax(dim = -1))

        output = torch.einsum("b h i j, b h j k -> b h i k", attention, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.W_o(output)

        return self.dropout(output)
    
class FeedForward(torch.nn.Module):
    """A feed forward layer for transformers.
    
    Parameters
    ----------
    dim : int
        The dimension of the input and output
    hidden_dim : int
        The dimension of the hidden layer
    dropout : float, optional
        The dropout rate, by default 0.
    activation : torch.nn.Module, optional
        The activation function, by default torch.nn.GELU"""
    def __init__(self, 
                 dim, 
                 hidden_dim, 
                 dropout = 0.,
                 activation = torch.nn.GELU):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(torch.nn.Module):
    """A residual transformer with attention and feed forward layers.
    
    Parameters
    ----------
    dim : int
        The dimension of the residual stream
    depth : int
        The number of attention and feed forward layers
    heads : int, optional
        The number of attention heads, by default 8
    head_dim : int, optional
        The dimension of the subspaces of the attention heads, by default 64
    dropout : float, optional
        The dropout rate, by default 0.
    context_x : int, optional
        The context length in the x-direction, by default 32
    context_y : int, optional
        The context length in the y-direction, by default None
    has_pos_emb : bool, optional
        Whether to use positional embeddings, by default True
    alibi : bool, optional
        Whether to use the alibi positional embedding, by default True
    """
    def __init__(self, 
                 dim, 
                 depth = 1, 
                 heads = 8,
                 head_dim = 64,
                 dropout = 0.,
                 context_x = 32,
                 context_y = None,
                 has_pos_emb = True,
                 alibi = True,
                 hidden_multiplier = 2):
        super().__init__()

        if context_y is not None:
            self.cross_attention = True
        else:
            self.cross_attention = False

        self.layers = torch.nn.ModuleList([])
        for i in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, 
                          n_heads = heads, 
                          dim_head = head_dim, 
                          dropout = dropout,
                          context_x = context_x,
                          context_y = context_y,
                          has_pos_emb = has_pos_emb,
                          alibi = alibi),
                FeedForward(dim, 
                            dim * hidden_multiplier, 
                            dropout = dropout)
            ]))

    def forward(self, x, y = None):
        for attention, ff in self.layers:
            x = x + attention(x, y = y)
            x = x + ff(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 100, 512).to(device)
    y = torch.randn(1, 100, 512).to(device)

    # test transformer
    transformer = Transformer(512, context_x = 100).to(device)

    # test regular transformer
    out = transformer(x)

    # test cross attention
    out = transformer(x, y = y)