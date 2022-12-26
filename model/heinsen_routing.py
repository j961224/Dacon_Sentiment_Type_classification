from __future__ import annotations
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import einsum

class Routing(nn.Module):
    """
    Routes input vectors to the output vectors that maximize "bang per bit"
    by best predicting them, with optimizations that reduce parameter count,
    memory use, and computation by orders of magnitude. Each vector is a
    capsule representing an entity in a context (e.g., a word in a paragraph,
    an object in an image). See "An Algorithm for Routing Vectors in
    Sequences" (Heinsen, 2022), https://arxiv.org/abs/2211.11754.
    Args:
        n_inp: int, number of input vectors. If -1, the number is variable.
        n_out: int, number of output vectors.
        d_inp: int, size of input vectors.
        d_out: int, size of output vectors.
        n_iters: (optional) int, number of iterations. Default: 2.
        normalize: (optional) bool, if True and d_out > 1, normalize each
            output vector's elements to mean 0 and variance 1. Default: True.
        memory_efficient: (optional) bool, if True, compute votes lazily to
            reduce memory use by O(n_inp * n_out * d_out), while increasing
            computation by only O(n_iters). Default: True.
        return_dict: (optional) bool, if True, return a dictionary with the
            final state of all internal and output tensors. Default: False.
    Input:
        x_inp: float tensor of input vectors [..., n_inp, d_inp].
    Output:
        x_out: float tensor of output vectors [..., n_out, d_out] by default,
            or a dict with output vectors as 'x_out' if return_dict is True.
    Sample usage:
        >>> # Route 100 vectors of size 1024 to 10 vectors of size 4096.
        >>> m = EfficientVectorRouting(n_inp=100, n_out=10, d_inp=1024, d_out=4096)
        >>> x_inp = torch.randn(100, 1024)  # 100 vectors of size 1024
        >>> x_out = m(x_inp)  # 10 vectors of size 4096
    """
    def __init__(self, config, num_labels: int, n_inp: int = 315, n_out: int = 3, d_inp: int = 512, d_out: int = 1, n_iters: int = 2,
                 normalize: bool = True, memory_efficient: bool = True, return_dict: bool = False) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        n_inp=315
        n_out=self.num_labels
        d_inp=config.hidden_size
        d_out=1
        assert n_inp > 0 or n_inp == -1, "Number of input vectors must be > 0 or -1 (variable)."
        assert n_out >= 2, "Number of output vectors must be at least 2."
        one_or_n_inp = max(1, n_inp)
        self.n_inp, self.n_out, self.d_inp, self.d_out, self.n_iters = (n_inp, n_out, d_inp, d_out, n_iters)
        self.normalize, self.memory_efficient, self.return_dict = (normalize, memory_efficient, return_dict)
        self.register_buffer('CONST_ones_over_n_out', torch.ones(n_out) / n_out)
        self.W_A = nn.Parameter(torch.empty(one_or_n_inp, d_inp).normal_(std=2.0 * d_inp**-0.5))
        self.B_A = nn.Parameter(torch.zeros(one_or_n_inp))
        self.W_F1 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
        self.W_F2 = nn.Parameter(torch.empty(d_inp, d_out).normal_(std=2.0 * d_inp**-0.5))
        self.B_F2 = nn.Parameter(torch.zeros(n_out, d_out))
        self.W_G1 = nn.Parameter(torch.empty(d_out, d_inp).normal_(std=d_out**-0.5))
        self.W_G2 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
        self.B_G2 = nn.Parameter(torch.zeros(n_out, d_inp))
        self.W_S = nn.Parameter(torch.empty(one_or_n_inp, n_out).normal_(std=d_inp**-0.5))
        self.B_S = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
        self.beta_use = nn.Parameter(torch.empty(n_inp, n_out).normal_())
        self.beta_ign = nn.Parameter(torch.empty(n_inp, n_out).normal_())
        self.N = nn.LayerNorm(d_out, elementwise_affine=False) if d_out > 1 else nn.Identity()
        self.f, self.log_f, self.softmax = (nn.Sigmoid(), nn.LogSigmoid(), nn.Softmax(dim=-1))

    def __repr__(self) -> str:
        cfg_str = ', '.join(f'{s}={getattr(self, s)}' for s in 'n_inp n_out d_inp d_out n_iters normalize memory_efficient return_dict'.split())
        return '{}({})'.format(self._get_name(), cfg_str)

    def forward(self, x_inp: torch.Tensor) -> Union[torch.Tensor, dict]:
        scaled_x_inp = x_inp * x_inp.shape[-2]**-0.5  # [...id]
        a_inp = (scaled_x_inp * self.W_A).sum(dim=-1) + self.B_A  # [...i]
        V = None if self.memory_efficient else einsum('...id,jd,dh->...ijh', scaled_x_inp, self.W_F1, self.W_F2) + self.B_F2
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = self.CONST_ones_over_n_out  # [j]
            else:
                pred_x_inp = einsum('...jh,hd,jd->...jd', self.N(x_out), self.W_G1, self.W_G2) + self.B_G2
                S = self.log_f(einsum('...id,...jd->...ij', x_inp, pred_x_inp) * self.W_S + self.B_S)
                R = self.softmax(S)  # [...ij]

            # D-step.
            D_use = f_a_inp * R  # [...ij]
            D_ign = f_a_inp - D_use  # [...ij]

            # M-step.
            phi = self.beta_use * D_use - self.beta_ign * D_ign  # [...ij] "bang per bit" coefficients
            x_out = einsum('...jd,jd,dh->...jh', einsum('...ij,...id->...jd', phi, scaled_x_inp), self.W_F1, self.W_F2) \
                + einsum('...ij,jh->...jh', phi, self.B_F2) if V is None else einsum('...ij,...ijh->...jh', phi, V)  # use precomputed V if available

        if self.normalize:
            x_out = self.N(x_out)
        
        return x_out.squeeze(-1)
        # if self.return_dict:
        #     return { 'a_inp': a_inp, 'V': V, 'pred_x_inp': pred_x_inp, 'S': S, 'R': R, 'D_use': D_use, 'D_ign': D_ign, 'phi': phi, 'x_out': x_out }
        # else:
        #     return x_out

# class Routing(nn.Module):
#     def __init__(
#         self, config, num_labels: int = 3
#     ):
#         super().__init__()
#         self.config = config
#         self.num_labels = num_labels
#         self.route = EfficientVectorRouting(n_inp=315, n_out=self.num_labels, d_inp=config.hidden_size, d_out=1).cuda() # d_out 4,3,3,2

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # n_inp, d_inp = (x.size(1), self.config.hidden_size)  # input seqs will have 1000 vectors of size 1024
#         # d_vec = self.num_labels # 4,3,3,2                # we will route each seq to a vector of size 2048

#         # model = EfficientVectorRouting(n_inp=n_inp, n_out=d_vec, d_inp=d_inp, d_out=1).cuda() # d_out 4,3,3,2

#         x_out = self.route(x).squeeze(-1)   # shape is [batch_sz, d_vec] -> x_out이 Type_logits
#         return x_out


















# class EfficientVectorRouting(nn.Module):
#     """
#     Routes input vectors to the output vectors that maximize "bang per bit"
#     by best predicting them, with optimizations that reduce parameter count,
#     memory use, and computation by orders of magnitude. Each vector is a
#     capsule representing an entity in a context (e.g., a word in a paragraph,
#     an object in an image). See "An Algorithm for Routing Vectors in
#     Sequences" (Heinsen, 2022), https://arxiv.org/abs/2211.11754.
#     Args:
#         n_inp: int, number of input vectors. If -1, the number is variable.
#         n_out: int, number of output vectors.
#         d_inp: int, size of input vectors.
#         d_out: int, size of output vectors.
#         n_iters: (optional) int, number of iterations. Default: 2.
#         normalize: (optional) bool, if True and d_out > 1, normalize each
#             output vector's elements to mean 0 and variance 1. Default: True.
#         memory_efficient: (optional) bool, if True, compute votes lazily to
#             reduce memory use by O(n_inp * n_out * d_out), while increasing
#             computation by only O(n_iters). Default: True.
#         return_dict: (optional) bool, if True, return a dictionary with the
#             final state of all internal and output tensors. Default: False.
#     Input:
#         x_inp: float tensor of input vectors [..., n_inp, d_inp].
#     Output:
#         x_out: float tensor of output vectors [..., n_out, d_out] by default,
#             or a dict with output vectors as 'x_out' if return_dict is True.
#     Sample usage:
#         >>> # Route 100 vectors of size 1024 to 10 vectors of size 4096.
#         >>> m = EfficientVectorRouting(n_inp=100, n_out=10, d_inp=1024, d_out=4096)
#         >>> x_inp = torch.randn(100, 1024)  # 100 vectors of size 1024
#         >>> x_out = m(x_inp)  # 10 vectors of size 4096
#     """
#     def __init__(self, n_inp: int, n_out: int, d_inp: int, d_out: int, n_iters: int = 2,
#                  normalize: bool = True, memory_efficient: bool = True, return_dict: bool = False) -> None:
#         super().__init__()
#         assert n_inp > 0 or n_inp == -1, "Number of input vectors must be > 0 or -1 (variable)."
#         assert n_out >= 2, "Number of output vectors must be at least 2."
#         one_or_n_inp = max(1, n_inp)
#         self.n_inp, self.n_out, self.d_inp, self.d_out, self.n_iters = (n_inp, n_out, d_inp, d_out, n_iters)
#         self.normalize, self.memory_efficient, self.return_dict = (normalize, memory_efficient, return_dict)
#         self.register_buffer('CONST_ones_over_n_out', torch.ones(n_out) / n_out)
#         self.W_A = nn.Parameter(torch.empty(one_or_n_inp, d_inp).normal_(std=2.0 * d_inp**-0.5))
#         self.B_A = nn.Parameter(torch.zeros(one_or_n_inp))
#         self.W_F1 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
#         self.W_F2 = nn.Parameter(torch.empty(d_inp, d_out).normal_(std=2.0 * d_inp**-0.5))
#         self.B_F2 = nn.Parameter(torch.zeros(n_out, d_out))
#         self.W_G1 = nn.Parameter(torch.empty(d_out, d_inp).normal_(std=d_out**-0.5))
#         self.W_G2 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
#         self.B_G2 = nn.Parameter(torch.zeros(n_out, d_inp))
#         self.W_S = nn.Parameter(torch.empty(one_or_n_inp, n_out).normal_(std=d_inp**-0.5))
#         self.B_S = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
#         self.beta_use = nn.Parameter(torch.empty(n_inp, n_out).normal_())
#         self.beta_ign = nn.Parameter(torch.empty(n_inp, n_out).normal_())
#         self.N = nn.LayerNorm(d_out, elementwise_affine=False) if d_out > 1 else nn.Identity()
#         self.f, self.log_f, self.softmax = (nn.Sigmoid(), nn.LogSigmoid(), nn.Softmax(dim=-1))

#     def __repr__(self) -> str:
#         cfg_str = ', '.join(f'{s}={getattr(self, s)}' for s in 'n_inp n_out d_inp d_out n_iters normalize memory_efficient return_dict'.split())
#         return '{}({})'.format(self._get_name(), cfg_str)

#     def forward(self, x_inp: torch.Tensor) -> Union[torch.Tensor, dict]:
#         scaled_x_inp = x_inp * x_inp.shape[-2]**-0.5  # [...id]
#         a_inp = (scaled_x_inp * self.W_A).sum(dim=-1) + self.B_A  # [...i]
#         V = None if self.memory_efficient else einsum('...id,jd,dh->...ijh', scaled_x_inp, self.W_F1, self.W_F2) + self.B_F2
#         f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
#         for iter_num in range(self.n_iters):

#             # E-step.
#             if iter_num == 0:
#                 R = self.CONST_ones_over_n_out  # [j]
#             else:
#                 pred_x_inp = einsum('...jh,hd,jd->...jd', self.N(x_out), self.W_G1, self.W_G2) + self.B_G2
#                 S = self.log_f(einsum('...id,...jd->...ij', x_inp, pred_x_inp) * self.W_S + self.B_S)
#                 R = self.softmax(S)  # [...ij]

#             # D-step.
#             D_use = f_a_inp * R  # [...ij]
#             D_ign = f_a_inp - D_use  # [...ij]

#             # M-step.
#             phi = self.beta_use * D_use - self.beta_ign * D_ign  # [...ij] "bang per bit" coefficients
#             x_out = einsum('...jd,jd,dh->...jh', einsum('...ij,...id->...jd', phi, scaled_x_inp), self.W_F1, self.W_F2) \
#                 + einsum('...ij,jh->...jh', phi, self.B_F2) if V is None else einsum('...ij,...ijh->...jh', phi, V)  # use precomputed V if available

#         if self.normalize:
#             x_out = self.N(x_out)
        
#         return x_out
#         # if self.return_dict:
#         #     return { 'a_inp': a_inp, 'V': V, 'pred_x_inp': pred_x_inp, 'S': S, 'R': R, 'D_use': D_use, 'D_ign': D_ign, 'phi': phi, 'x_out': x_out }
#         # else:
#         #     return x_out

# class Routing(nn.Module):
#     def __init__(
#         self, config, num_labels: int = 3
#     ):
#         super().__init__()
#         self.config = config
#         self.num_labels = num_labels
#         self.route = EfficientVectorRouting(n_inp=315, n_out=self.num_labels, d_inp=config.hidden_size, d_out=1).cuda() # d_out 4,3,3,2

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # n_inp, d_inp = (x.size(1), self.config.hidden_size)  # input seqs will have 1000 vectors of size 1024
#         # d_vec = self.num_labels # 4,3,3,2                # we will route each seq to a vector of size 2048

#         # model = EfficientVectorRouting(n_inp=n_inp, n_out=d_vec, d_inp=d_inp, d_out=1).cuda() # d_out 4,3,3,2

#         x_out = self.route(x).squeeze(-1)   # shape is [batch_sz, d_vec] -> x_out이 Type_logits
#         return x_out