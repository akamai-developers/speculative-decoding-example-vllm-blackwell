import os
import json
import time
import requests


def run_inference(url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    start = time.time()
    first_token_time = None
    output_text = ""
    usage = None

    response = requests.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.environ.get("TEMPERATURE", "0.1")),
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
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
        
        if chunk.get("usage") is not None:
            usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        token = delta.get("content", "")

        if token:
            if first_token_time is None:
                first_token_time = time.time()
            output_text += token

    end = time.time()

    total_latency = end - start
    ttft = first_token_time - start if first_token_time else None

    completion_tokens = None

    if usage is not None:
        completion_tokens = usage.get("completion_tokens")

    if completion_tokens is None:
        # Fallback only if vLLM does not return usage.
        completion_tokens = len(output_text.split())

    output_tps = completion_tokens / total_latency if total_latency > 0 else 0

    return {
        "text": output_text,
        "latency": total_latency,
        "ttft": ttft,
        "output_tokens": completion_tokens,
        "output_tps": output_tps,
    }