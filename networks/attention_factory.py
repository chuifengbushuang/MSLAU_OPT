from networks.crossformer_attention import CrossFormerLSDAttention
from networks.msla import MSLA


def build_token_attention(attn_type, dim, num_heads, **kwargs):
    attn_type = (attn_type or "msla").lower()

    if attn_type == "msla":
        return MSLA(dim=dim, num_heads=num_heads, **kwargs)

    if attn_type in {"crossformer", "crossformer_lsda"}:
        return CrossFormerLSDAttention(dim=dim, num_heads=num_heads, **kwargs)

    raise ValueError(f"Unsupported attention type: {attn_type}")
