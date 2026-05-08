from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int) -> int:
    """Divides two integers"""
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def local_kv_heads(num_kv_heads: int, tp_size: int) -> int:
    """Return per-rank KV heads, replicating heads when tp_size > num_kv_heads."""
    if num_kv_heads >= tp_size:
        return div_even(num_kv_heads, tp_size)
    assert tp_size % num_kv_heads == 0, (
        f"{tp_size = } must be divisible by {num_kv_heads = } when KV heads are replicated"
    )
    return 1


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()
