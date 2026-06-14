from __future__ import annotations

import argparse
import json
import time

import torch

from minisgl.kernel.activation_quant import per_token_quant_int8_triton


def bench(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 32, 128, 1024])
    parser.add_argument("--cols", type=int, nargs="+", default=[2048])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    results: list[dict[str, float | int | str]] = []

    for rows in args.rows:
        for cols in args.cols:
            x = torch.randn(rows, cols, device="cuda", dtype=dtype).contiguous()
            x_q = torch.empty_like(x, dtype=torch.int8)
            x_s = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
            ms = bench(
                lambda: per_token_quant_int8_triton(x, x_q, x_s),
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "dtype": args.dtype,
                    "ms": ms,
                    "us_per_row": ms * 1000.0 / rows,
                }
            )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
