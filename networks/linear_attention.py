import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):

    def __init__(self, dim, num_heads, linear_attn_type="legacy", eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.linear_attn_type = linear_attn_type.lower()
        self.eps = eps

        if self.linear_attn_type not in {"legacy", "relu", "elu"}:
            raise ValueError(f"Unsupported linear_attn_type: {linear_attn_type}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def _apply_kernel(self, x):
        if self.linear_attn_type == "relu":
            return F.relu(x) + self.eps
        if self.linear_attn_type == "elu":
            return F.elu(x) + 1.0 + self.eps
        return x

    def _legacy_attention(self, q, k, v):
        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        return query @ context

    def _kernel_attention(self, q, k, v):
        q = self._apply_kernel(q)
        k = self._apply_kernel(k)

        context = k.transpose(-2, -1) @ v
        normalizer = (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True)
        normalizer = normalizer.clamp_min(self.eps)
        return (q @ context) / normalizer

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.linear_attn_type == "legacy":
            x = self._legacy_attention(q, k, v)
        else:
            x = self._kernel_attention(q, k, v)
        x = x.reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x
