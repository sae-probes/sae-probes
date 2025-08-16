from __future__ import annotations


def get_layer_from_hook_name(hook_name: str) -> int | None:
    """Extract the transformer block index from a TransformerLens hook name.

    Examples:
    - "blocks.12.hook_resid_post" -> 12
    - "blocks.0.hook_attn_out" -> 0
    - "hook_embed" -> None (no block index)
    - "blocks.not_an_int.hook_resid_post" -> None
    """
    if not hook_name:
        return None
    parts: list[str] = hook_name.split(".")
    if len(parts) < 2:
        return None
    if parts[0] != "blocks":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None
