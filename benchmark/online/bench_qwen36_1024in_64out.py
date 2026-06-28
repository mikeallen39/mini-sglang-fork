from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from transformers import AutoTokenizer


@dataclass(frozen=True)
class RunResult:
    ttft_ms: float
    e2e_s: float
    output_tokens: int
    output_text: str


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    user_content_tokens: int
    final_prompt_tokens: int


def load_prompt_from_file(tokenizer, path: str, n: int) -> str:
    text = Path(path).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < n:
        repeated: list[int] = []
        while len(repeated) < n:
            repeated.extend(token_ids)
        token_ids = repeated[:n]
    else:
        token_ids = token_ids[:n]
    return tokenizer.decode(token_ids)


def _load_token_pool(tokenizer, path: str, n: int) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Prompt file is empty after tokenization: {path}")
    repeated: list[int] = []
    target = max(n + 4096, n * 2)
    while len(repeated) < target:
        repeated.extend(token_ids)
    return repeated


def _render_chat_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    assert isinstance(rendered, str)
    return rendered


def _count_final_prompt_tokens(tokenizer, prompt: str, enable_thinking: bool) -> int:
    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def build_prompt(
    tokenizer,
    path: str,
    input_tokens: int,
    enable_thinking: bool,
    input_token_mode: str,
) -> PromptSpec:
    if input_token_mode == "raw-content":
        prompt = load_prompt_from_file(tokenizer, path, input_tokens)
        final_prompt_tokens = _count_final_prompt_tokens(tokenizer, prompt, enable_thinking)
        user_content_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        return PromptSpec(
            prompt=prompt,
            user_content_tokens=user_content_tokens,
            final_prompt_tokens=final_prompt_tokens,
        )

    if input_token_mode != "final-chat":
        raise ValueError(f"Unsupported input token mode: {input_token_mode}")

    token_pool = _load_token_pool(tokenizer, path, input_tokens)
    lo, hi = 0, len(token_pool)
    best_len = None
    best_prompt = None
    best_final_tokens = None

    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = tokenizer.decode(token_pool[:mid])
        final_prompt_tokens = _count_final_prompt_tokens(tokenizer, prompt, enable_thinking)
        if final_prompt_tokens <= input_tokens:
            best_len = mid
            best_prompt = prompt
            best_final_tokens = final_prompt_tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_len is None or best_prompt is None or best_final_tokens is None:
        raise RuntimeError("Failed to find a prompt prefix within the requested token budget")

    if best_final_tokens != input_tokens:
        scan_end = min(len(token_pool), best_len + 512)
        for prefix_len in range(best_len + 1, scan_end + 1):
            prompt = tokenizer.decode(token_pool[:prefix_len])
            final_prompt_tokens = _count_final_prompt_tokens(tokenizer, prompt, enable_thinking)
            if final_prompt_tokens == input_tokens:
                best_len = prefix_len
                best_prompt = prompt
                best_final_tokens = final_prompt_tokens
                break
            if final_prompt_tokens > input_tokens:
                break

    if best_final_tokens != input_tokens:
        raise RuntimeError(
            "Unable to construct a prompt whose final chat-formatted length "
            f"is exactly {input_tokens} tokens; nearest value is {best_final_tokens}."
        )

    user_content_tokens = len(tokenizer.encode(best_prompt, add_special_tokens=False))
    return PromptSpec(
        prompt=best_prompt,
        user_content_tokens=user_content_tokens,
        final_prompt_tokens=best_final_tokens,
    )


def measure_one(
    base_url: str,
    model: str,
    prompt: str,
    output_len: int,
    tokenizer,
    enable_thinking: bool,
) -> RunResult:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    start = time.perf_counter()
    first_token_at = None
    pieces: List[str] = []

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=1800) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            token_text = delta.get("content") or delta.get("reasoning_content") or ""
            if token_text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(token_text)

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No output token received from server")

    output_text = "".join(pieces)
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    return RunResult(
        ttft_ms=(first_token_at - start) * 1000,
        e2e_s=end - start,
        output_tokens=output_tokens,
        output_text=output_text,
    )


def summarize_runs(results: List[RunResult]) -> None:
    if len(results) != 5:
        raise ValueError(f"Expected exactly 5 runs, got {len(results)}")

    for i, result in enumerate(results, start=1):
        print(
            f"run{i}: "
            f"TTFT={result.ttft_ms:.2f} ms, "
            f"E2E={result.e2e_s:.4f} s, "
            f"output_tokens={result.output_tokens}"
        )

    steady = results[1:]
    steady_ttft = [r.ttft_ms for r in steady]
    steady_e2e = [r.e2e_s for r in steady]
    steady_output_tokens = [r.output_tokens for r in steady]
    total_output_tokens = sum(steady_output_tokens)
    total_wall_time = sum(steady_e2e)
    output_tps = total_output_tokens / total_wall_time if total_wall_time > 0 else 0.0
    avg_ms_per_output_token = (
        (total_wall_time / total_output_tokens) * 1000 if total_output_tokens > 0 else 0.0
    )

    print("")
    print("[STEADY_STATE]")
    print(
        f"run2-run5 avg TTFT={statistics.mean(steady_ttft):.2f} ms, "
        f"avg E2E={statistics.mean(steady_e2e):.4f} s"
    )
    print(
        f"run2-run5 output_tps={output_tps:.2f} tok/s, "
        f"avg_ms_per_output_token={avg_ms_per_output_token:.2f} ms, "
        f"avg_output_tokens={statistics.mean(steady_output_tokens):.2f}"
    )


def save_run1_output(path: str, result: RunResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"TTFT={result.ttft_ms:.2f} ms",
                f"E2E={result.e2e_s:.4f} s",
                f"output_tokens={result.output_tokens}",
                "",
                result.output_text,
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Qwen3.6 online benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", default="/mnt/82_store/LLM-weights/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--prompt-file",
        default="benchmark/prompts/qwen36_1024in_science_long_prompt.txt",
    )
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument(
        "--input-token-mode",
        choices=["final-chat", "raw-content"],
        default="final-chat",
        help=(
            "How to interpret --input-tokens. "
            "'final-chat' counts the fully chat-templated prompt sent to the model; "
            "'raw-content' preserves the previous behavior and counts only the user content."
        ),
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--run1-output-file",
        default="my_docs/results/latest_run1_output.txt",
    )
    args = parser.parse_args()

    if args.runs != 5:
        raise ValueError("Unified benchmark requires --runs 5")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_spec = build_prompt(
        tokenizer=tokenizer,
        path=args.prompt_file,
        input_tokens=args.input_tokens,
        enable_thinking=args.enable_thinking,
        input_token_mode=args.input_token_mode,
    )

    print(f"model={args.model}")
    print(f"base_url={args.base_url}")
    print(f"prompt_file={args.prompt_file}")
    print(f"input_tokens={args.input_tokens}")
    print(f"input_token_mode={args.input_token_mode}")
    print(f"user_content_tokens={prompt_spec.user_content_tokens}")
    print(f"final_prompt_tokens={prompt_spec.final_prompt_tokens}")
    print(f"output_tokens={args.output_tokens}")
    print(f"runs={args.runs}")
    print(f"enable_thinking={args.enable_thinking}")
    print(f"run1_output_file={args.run1_output_file}")
    print("")

    results = [
        measure_one(
            args.base_url,
            args.model,
            prompt_spec.prompt,
            args.output_tokens,
            tokenizer,
            args.enable_thinking,
        )
        for _ in range(args.runs)
    ]
    save_run1_output(args.run1_output_file, results[0])
    summarize_runs(results)


if __name__ == "__main__":
    main()
