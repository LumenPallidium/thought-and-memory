import torch
import numpy as np
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
                 autoregressive = True,
                 max_recurrent_steps = 8):
        """
        ArsMemoria model, a transformer with a memory encoder and a predictor.
        The predictor is autoregressive, with recurrent memory tokens. Training
        is made more compatible with backpropagation by simplifying the recurrence.

        Parameters
        ----------
        dim : int
            Dimension of the model residual stream
        embed_dim : int
            Dimension of the token embedding
        n_heads : int
            Number of attention heads
        dropout : float
            Dropout rate
        predictor_depth : int
            Depth of the predictor transformer
        predictor_context : int
            Number of tokens to use for the predictor context
        memory_depth : int
            Depth of the memory encoder transformer
        memory_context : int
            Number of tokens to use for the memory context
        autoregressive : bool
            Whether to use autoregressive masking in attention
        max_recurrent_steps : int
            For the memory loss, max number of chunks of predictor tokens to use
            (= number of recurrent steps)        
        """
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        # TODO: constant for each model cause i may add head-specific memory dynamics
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.predictor_context = predictor_context
        self.memory_context = memory_context
        self.autoregressive = autoregressive
        self.max_recurrent_steps = max_recurrent_steps

        self.dropout = dropout

        self.embedder = torch.nn.Embedding(embed_dim, dim)
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

        if autoregressive:
            self._init_mask()
        else:
            self.mask = None

    def _init_mask(self):
        """Generate the mask for autoregressive prediction. Memory tokens can't be influenced
        by future tokens, but can influence themselves and future tokens."""
        # standard autoregressive mask
        mask = torch.full((self.predictor_context, self.predictor_context), float("-inf"))
        mask = torch.triu(mask, diagonal = 1)

        # memory tokens cannot be modified by future
        mask_top = torch.full((self.memory_context, self.predictor_context), float("-inf"))
        mask = torch.cat([mask_top, mask], dim = 0)

        # memory tokens can influence themselves and later tokens
        mask_left = torch.zeros((self.predictor_context + self.memory_context, self.memory_context))
        mask = torch.cat([mask_left, mask], dim = 1)
        self.register_buffer("mask", mask)
    
    def embed_step(self, x, memories = None):
        """
        Embed input tensor and (optionally) memories.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D)
        memories : torch.Tensor
            Memory tensor of shape (B, M, D), where M is the number of memory tokens
        
        Returns
        -------
        embed : torch.Tensor
            Embedded tensor of shape (B, L, D)
        memories : torch.Tensor
            Embedded memory tensor of shape (B, M, D)
        """
        if memories is None:
            memories = self.memory_token.repeat(x.shape[0], 1, 1)
        full_x = torch.cat([memories.clone(), x], dim = 1)
        embed = self.predictor(full_x, mask = self.mask)[:, self.memory_context:, :]
        return embed, memories

    def forward(self, x, memories = None):
        """
        Run the input and memories through both the predictor and memory encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D)
        memories : torch.Tensor
            Memory tensor of shape (B, M, D), where M is the number of memory tokens
        
        Returns
        -------
        logits : torch.Tensor
            Logits tensor of shape (B, L, D)
        embed : torch.Tensor
            Embedded tensor of shape (B, L, D)
        memories : torch.Tensor
            Embedded memory tensor of shape (B, M, D)
        """
        embed, memories = self.embed_step(x, memories = memories)

        new_memories = self.memory_encoder(memories, 
                                           y = embed)
        logits = self.decoder(embed)
        return logits, embed, new_memories
    
    def sample(self, prompt_indices, memories = None, len = 20, top_k = 5):
        for i in range(len):
            encoded = self.embedder(prompt_indices)
            logits, _, memories = self(encoded, memories = memories)
            logits = logits[:, -1, :]
            logits = torch.nn.functional.softmax(logits, dim = -1)
            logits = torch.multinomial(logits, 1)
            prompt_indices = torch.cat([prompt_indices, logits], dim = 1)
        return prompt_indices

    def autoregressive_loss(self, 
                            embedded_labels, 
                            labels, 
                            memories = None):
        """Compute standard autoregressive loss."""
        logits, embed, new_memories = self(embedded_labels, memories = memories)
        # shift labels
        labels = labels[:, 1:]
        logits = logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1),
                                                 labels)

        return embed, new_memories, loss
    
    #TODO : this only uses a single past memory/embedding, can we do multiple?
    #TODO : MSE between token running average is an interesting idea for this
    def recall_loss(self,
                    embedded_labels,
                    memories):
        """Recall loss, the abiity of the predictor to reconstruct 
        the input from the memory tokens."""
        # remove BOS
        embedded_labels = embedded_labels[:, 1:, :]
        masked_seq = self.mask_token.repeat(embedded_labels.shape[0],
                                            embedded_labels.shape[1] + 1,
                                            1)
        masked_seq[:, 0, :] = self.recall_token
        _, embed, _ = self(masked_seq, memories = memories)
        loss = torch.nn.functional.mse_loss(embed[:, 1:, :],
                                            embedded_labels)
        return loss
    
    def memory_loss(self, 
                    embed, 
                    memories,
                    run_vicreg = True,
                    vicreg_samples = 4):
        """
        The only recurrent loss, it only acts on the memory encoder.
        Essentially, run a single embedding through the memory network,
        then run a chunked version of it through recurrently. The loss
        ensures they are similar, i.e. doing a large chunk of tokens at once
        or many small chunks gets the same result.

        Can run VICReg where the different memory encodings are treated
        as two different "views" of the memory.
        """
        with torch.no_grad():
            full_memory = self.memory_encoder(memories, y = embed)

        # generate splits randomly, with varying number and size
        n_splits = np.random.randint(2, self.max_recurrent_steps)
        split_indices = torch.rand(n_splits).sort()[0]
        split_indices = (split_indices * self.predictor_context).long()

        # recurrent chunk estimation
        embed_splits = torch.tensor_split(embed, split_indices, dim = 1)
        for embed_split in embed_splits:
            memories = self.memory_encoder(memories, y = embed_split)

        # make them close to encourage compositionality
        loss = torch.nn.functional.mse_loss(memories, full_memory)
        if run_vicreg:
            cat_memories = torch.cat([memories, full_memory], dim = 1)
            indices = torch.randint(0, cat_memories.shape[1], (vicreg_samples,)).to(cat_memories.device)
            loss += vicreg(cat_memories[:, indices, :].view(-1, self.dim))
        return loss

#TODO : interesting https://www.biorxiv.org/content/10.1101/2022.11.04.515143v2
def vicreg(embed, var_weight = 1, cov_weight = 0.001, gamma = 4, eps = 1e-5):
    """
    The variance and covariance part of VICReg, 
    adapted from here:
    
    https://arxiv.org/pdf/2105.04906.pdf
    """
    embed_mean = embed.mean(dim = 0, keepdim = True)

    # variance term
    variance = gamma - (embed.var(dim = 0) + eps).sqrt()
    variance = torch.nn.functional.relu(variance).mean()

    # covariance term
    # (d x batch) @ (batch x d) = (d x d)
    covariance_context = (embed - embed_mean).T @ (embed - embed_mean) / embed.shape[0]

    # second vicreg term
    covariance = covariance_context.triu().pow(2).sum()
    covariance /= covariance_context.shape[0]
    loss = var_weight * variance + cov_weight * covariance
    return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_indices = torch.randint(2, 7, (1, 254)).to(device)

    bos = torch.tensor([0]).to(device)
    eos = torch.tensor([1]).to(device)

    random_indices = torch.cat([bos.repeat(random_indices.shape[0], 1),
                                random_indices,
                                eos.repeat(random_indices.shape[0], 1)], dim = 1)


    ars = ArsMemoria(embed_dim = 7).to(device)
    optimizer = torch.optim.Adam(ars.parameters(), lr = 1e-4)

    embedded_indices = ars.embedder(random_indices)

    memories = torch.randn(1, 16, 512).to(device)
    # test first pass
    embed, memories, ar_loss = ars.autoregressive_loss(embedded_indices, 
                                                       random_indices,
                                                       memories = memories)
    ar_loss.backward()
    # TODO : maybe make a detach + clone wrapper
    recall_loss = ars.recall_loss(embed.detach().clone().requires_grad_(True), 
                                  memories.detach().clone().requires_grad_(True))
    recall_loss.backward()
    optimizer.step()

    memory_loss = ars.memory_loss(embed.detach().clone().requires_grad_(True), 
                                  memories.detach().clone().requires_grad_(True))
    memory_loss.backward()
    optimizer.step()

