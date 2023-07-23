import math

import torch
import gpytorch


class KernelAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 kernel_class: gpytorch.kernels.Kernel,
                 num_classes: int,
                 lmbda: float = 0.1,
                 inverse_function=None
                 ):
        super().__init__()
        self.kernel = kernel_class.initialize()
        self.lmbda = torch.nn.Parameter(torch.tensor(lmbda), requires_grad=False)
        self.input_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.out_projection = torch.nn.Linear(embed_dim, num_classes)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.inverse_function = inverse_function

    def forward(self, x):
        bs, seq_len, f = x.shape

        input_projection = self.input_projection(x)
        input_projection = input_projection.reshape(bs, seq_len, self.n_heads, self.head_dim)
        input_projection = input_projection.permute(0, 2, 1, 3)

        k = self.kernel(input_projection, input_projection).evaluate()
        k_inversed = self.inverse_function(k - torch.eye(seq_len, device=x.device) * self.lmbda)
        attention = k @ k_inversed

        output = attention @ input_projection
        out_projection = self.out_projection(
            output.reshape(bs, seq_len, self.embed_dim)
        )

        return out_projection


class LinearAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.input_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.out_projection = torch.nn.Linear(embed_dim, num_classes)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

    def forward(self, x):
        bs, seq_len, f = x.shape

        input_projection = self.input_projection(x)
        input_projection = input_projection.reshape(bs, seq_len, self.n_heads, self.head_dim)
        input_projection = input_projection.permute(0, 2, 1, 3)

        attention = torch.matmul(input_projection, input_projection.transpose(-2, -1)) / math.sqrt(f)

        output = attention @ input_projection
        out_projection = self.out_projection(
            output.reshape(bs, seq_len, self.embed_dim)
        )

        return out_projection


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = torch.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class LinearTransformerLayer(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_classes):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = torch.nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        return self.o_proj(values)