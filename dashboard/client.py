import os
import json
import time
import requests


def run_streaming_vllm(url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    start = time.time()
    first_token_time = None
    output_text = ""

    response = requests.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": os.environ["TEMPERATURE"],
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=180,
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")

        if decoded.startswith("data: "):
            decoded = decoded[len("data: "):]

        if decoded.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(decoded)
        except json.JSONDecodeError:
            continue

        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")

        if token:
            if first_token_time is None:
                first_token_time = time.time()
            output_text += token

    end = time.time()

    total_latency = end - start
    ttft = first_token_time - start if first_token_time else None

    # Simple approximation. Later you can replace this with tokenizer-based counting.
    output_tokens = len(output_text.split())
    output_tps = output_tokens / total_latency if total_latency > 0 else 0

    return {
        "text": output_text,
        "latency": total_latency,
        "ttft": ttft,
        "output_tokens": output_tokens,
        "output_tps": output_tps,
    }