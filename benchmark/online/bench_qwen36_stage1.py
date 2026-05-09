from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import List

import requests
from transformers import AutoTokenizer


@dataclass(frozen=True)
class CaseSpec:
    name: str
    input_len: int
    output_len: int
    repeats: int


@dataclass(frozen=True)
class RunResult:
    ttft_ms: float
    e2e_s: float
    output_tokens: int


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    idx = min(int(len(sorted_values) * q), len(sorted_values) - 1)
    return sorted_values[idx]


def generate_prompt(tokenizer, n: int) -> str:
    base = "请简洁回答以下问题，并保持内容稳定可复现："
    token_ids = tokenizer.encode(base, add_special_tokens=False)
    filler = "性能测试"
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    while len(token_ids) < n:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:n]
    return tokenizer.decode(token_ids)


def measure_one(base_url: str, model: str, prompt: str, output_len: int, tokenizer) -> RunResult:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    start = time.perf_counter()
    first_token_at = None
    pieces: List[str] = []

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(content)

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No output token received from server")

    output_text = "".join(pieces)
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    return RunResult(
        ttft_ms=(first_token_at - start) * 1000,
        e2e_s=end - start,
        output_tokens=output_tokens,
    )


def summarize_case(case: CaseSpec, results: List[RunResult]) -> None:
    ttft_ms = sorted(result.ttft_ms for result in results)
    e2e_s = sorted(result.e2e_s for result in results)
    total_output_tokens = sum(result.output_tokens for result in results)
    total_wall_time = sum(result.e2e_s for result in results)
    output_tps = total_output_tokens / total_wall_time if total_wall_time > 0 else 0.0
    avg_tokens = statistics.mean(result.output_tokens for result in results)
    avg_tpot_ms = ((total_wall_time / total_output_tokens) * 1000) if total_output_tokens > 0 else 0.0

    print(f"[CASE] {case.name}")
    print(f"input_len={case.input_len}, output_len={case.output_len}, repeats={case.repeats}")
    print(
        f"ttft: avg={statistics.mean(ttft_ms):.2f}ms, "
        f"p50={percentile(ttft_ms, 0.5):.2f}ms, "
        f"p90={percentile(ttft_ms, 0.9):.2f}ms, "
        f"max={ttft_ms[-1]:.2f}ms"
    )
    print(
        f"e2e: avg={statistics.mean(e2e_s):.2f}s, "
        f"p50={percentile(e2e_s, 0.5):.2f}s, "
        f"p90={percentile(e2e_s, 0.9):.2f}s, "
        f"max={e2e_s[-1]:.2f}s"
    )
    print(
        f"throughput: output_tokens={total_output_tokens}, avg_output_tokens={avg_tokens:.2f}, "
        f"output_tps={output_tps:.2f} tok/s, approx_tpot={avg_tpot_ms:.2f}ms"
    )
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 stage1 online benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:1921/v1")
    parser.add_argument("--model", default="/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = [
        CaseSpec(name="short_prefill_short_decode", input_len=64, output_len=32, repeats=3),
        CaseSpec(name="medium_prefill_short_decode", input_len=256, output_len=32, repeats=1),
    ]

    print(f"model={args.model}")
    print(f"base_url={args.base_url}")
    print("")

    warmup_prompt = "请简短回答：1+1等于几？"
    warmup = measure_one(args.base_url, args.model, warmup_prompt, 16, tokenizer)
    print(
        f"[WARMUP] ttft={warmup.ttft_ms:.2f}ms, "
        f"e2e={warmup.e2e_s:.2f}s, output_tokens={warmup.output_tokens}"
    )
    print("")

    for case in cases:
        prompt = generate_prompt(tokenizer, case.input_len)
        results = [
            measure_one(args.base_url, args.model, prompt, case.output_len, tokenizer)
            for _ in range(case.repeats)
        ]
        summarize_case(case, results)


if __name__ == "__main__":
    main()
