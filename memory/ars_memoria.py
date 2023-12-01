import torch
from transformer_blocks import Transformer

class ArsMemoria(torch.nn.Module):
    def __init__(self,
                 dim = 512,
                 n_heads = 8,
                 dropout = 0.4,
                 predictor_depth = 8,
                 predictor_context = 256,
                 memory_depth = 4,
                 memory_context = 16):
        super().__init__()
        self.dim = dim
        # TODO: constant for each model cause i may add head-specific memory dynamics
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.predictor_context = predictor_context
        self.memory_context = memory_context

        self.dropout = dropout

        self.predictor = Transformer(dim = dim,
                                     depth = predictor_depth,
                                     heads = n_heads,
                                     head_dim = self.head_dim,
                                     dropout = dropout,
                                     context_x = predictor_context + memory_context)
        self.memory_encoder = Transformer(dim = dim,
                                          depth = memory_depth,
                                          heads = n_heads,
                                          head_dim = self.head_dim,
                                          dropout = dropout,
                                          context_x = memory_context,
                                          context_y = predictor_context)
        self.mask_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.memory_token = torch.nn.Parameter(torch.randn(1, memory_context, dim))
        
    def forward(self, x, memories = None):
        if memories is None:
            memories = self.memory_token.repeat(x.shape[0], 1, 1)
        full_x = torch.cat([x, memories], dim = 1)
        y = self.predictor(full_x)[:, :self.predictor_context, :]

        new_memories = self.memory_encoder(memories, y = y)
        return y, new_memories
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 256, 512).to(device)

    ars = ArsMemoria().to(device)

    # test first pass
    y, memories = ars(x)

    x = torch.randn(1, 256, 512).to(device)
    # test second pass, with memories
    y, memories = ars(x, memories = memories)

