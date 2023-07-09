import torch
import gpytorch


class KernelAttention(torch.nn.Module):
    def __init__(self,
                 kernel_class: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel(),
                 lmbda: float = 0.1
                 ):
        super().__init__()
        self.kernel = kernel_class.initialize()
        self.lmbda = torch.nn.Parameter(torch.tensor(lmbda), requires_grad=False)

    def forward(self, x):
        bs, seq_len, f = x.shape

        K = self.kernel(x, x).evaluate()
        attention = K @ (
                torch.inverse(K - torch.eye(seq_len, device=x.device) * self.lmbda)
        )
        output = attention @ x
        return output
