import torch
from transformer_blocks import Transformer

class ArsMemoria(torch.nn.Module):
    def __init__(self,
                 dim = 512,
                 embed_dim = 512,
                 n_heads = 8,
                 dropout = 0.4,
                 predictor_depth = 8,
                 predictor_context = 256,
                 memory_depth = 4,
                 memory_context = 16,
                 autoregressive = True):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
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
        self.decoder = torch.nn.Linear(dim, embed_dim)
        
        self.mask_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.memory_token = torch.nn.Parameter(torch.randn(1, memory_context, dim))
        # token prefix when doing recall task
        self.recall_token = torch.nn.Parameter(torch.randn(1, 1, dim))

        self.mask = self._init_mask(autoregressive)

    def _init_mask(self, autoregressive):
        if autoregressive:
            # standard autoregressive mask
            mask = torch.full((self.predictor_context, self.predictor_context), float("-inf"))
            mask = torch.triu(mask, diagonal = 1)

            # memory tokens cannot be modified by future
            mask_top = torch.full((self.memory_context, self.predictor_context), float("-inf"))
            mask = torch.cat([mask_top, mask], dim = 0)

            # memory tokens can influence themselves and later tokens
            mask_left = torch.zeros((self.predictor_context + self.memory_context, self.memory_context))
            mask = torch.cat([mask_left, mask], dim = 1)
        else:
            mask = None
        return mask
    
    def autoregression_step(self, x, memories = None):
        if memories is None:
            memories = self.memory_token.repeat(x.shape[0], 1, 1)
        full_x = torch.cat([memories.clone(), x], dim = 1)
        y = self.predictor(full_x, mask = self.mask)[:, self.memory_context:, :]
        return y, memories

    def forward(self, x, memories = None):
        y, memories = self.autoregression_step(x, memories = memories)

        new_memories = self.memory_encoder(memories.detach().requires_grad_(), 
                                           y = y)
        logits = self.decoder(y)
        return logits, y, new_memories
    
    def autoregressive_loss(self, 
                            embedded_labels, 
                            labels, 
                            memories = None,
                            recall_loss = True):
        logits, y, new_memories = self(embedded_labels, memories = memories)
        # remove BOS and EOS
        logits = logits[:, 1:-1, :]
        # labels have no BOS and EOS
        loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1),
                                                 labels)
        if recall_loss:
            recall_loss = self.recall_loss(embedded_labels, memories)
            loss += recall_loss
        return y, new_memories, loss
    
    def recall_loss(self, embedded_labels, memories):
        # remove BOS
        embedded_labels = embedded_labels[:, 1:, :]
        masked_seq = self.mask_token.repeat(embedded_labels.shape[0],
                                            embedded_labels.shape[1] + 1,
                                            1)
        masked_seq[:, 0, :] = self.recall_token
        logits, _, _ = self(masked_seq, memories = memories)
        loss = torch.nn.functional.mse_loss(logits[:, 1:, :],
                                            embedded_labels)
        return loss
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_indices = torch.randint(0, 5, (1, 254)).to(device)

    embedder = torch.nn.Embedding(5, 512).to(device)
    bos = torch.randn(1, 1, 512).to(device)
    eos = torch.randn(1, 1, 512).to(device)

    embedded_indices = embedder(random_indices)
    embedded_labels = torch.cat([bos, embedded_indices, eos], dim = 1)

    ars = ArsMemoria().to(device)

    # test first pass
    y, memories, loss = ars.autoregressive_loss(embedded_labels, random_indices)

    # test second pass, with memories
    y, memories, loss = ars.autoregressive_loss(embedded_labels, random_indices, memories = memories)

