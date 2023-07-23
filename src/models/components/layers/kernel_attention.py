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
