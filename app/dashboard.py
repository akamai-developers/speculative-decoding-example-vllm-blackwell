import os
import time
import queue
import threading
import requests
import streamlit as st
from openai import OpenAI

BASELINE_URL = "http://127.0.0.1:8000/v1"
SPEC_URL = "http://127.0.0.1:8001/v1"
PROMETHEUS_URL = "http://127.0.0.1:9090"

TARGET_MODEL = os.getenv("TARGET_MODEL")

baseline_client = OpenAI(base_url=BASELINE_URL, api_key="EMPTY")
spec_client = OpenAI(base_url=SPEC_URL, api_key="EMPTY")

PROMPTS = {
    "Simple explanation": "Explain speculative decoding in 5 beginner-friendly bullet points.",
    "Long generation": "Write a 500-word explanation of how vLLM serves requests, including batching, KV cache, and token generation.",
    "Harder reasoning": "Given a Python service that gets slower as concurrent requests increase, explain three possible bottlenecks and how you would debug them.",
}

st.set_page_config(page_title="vLLM Speculative Decoding Demo", layout="wide")
st.title("vLLM Baseline vs Speculative Decoding")

preset = st.radio("Prompt preset", list(PROMPTS.keys()), horizontal=True)
prompt = st.text_area("Prompt", value=PROMPTS[preset], height=130)

max_tokens = st.slider(
    "Max generated tokens",
    min_value=64,
    max_value=1024,
    value=512,
    step=64,
)

temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)


def stream_request(name, client, model, prompt, out_queue):
    start = time.perf_counter()
    text = ""
    chunk_count = 0
    first_token_time = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""

            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                    out_queue.put((name, "ttft", first_token_time))

                text += delta
                chunk_count += 1
                out_queue.put((name, "token", delta))

        total_latency = time.perf_counter() - start

        out_queue.put(
            (
                name,
                "done",
                {
                    "text": text,
                    "latency": total_latency,
                    "ttft": first_token_time,
                    "chunks": chunk_count,
                    "chunks_per_second": chunk_count / total_latency if total_latency > 0 else 0,
                },
            )
        )

    except Exception as e:
        out_queue.put((name, "error", str(e)))


def query_prometheus(query):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=3,
        )
        r.raise_for_status()
        data = r.json()
        result = data["data"]["result"]

        if not result:
            return None

        return float(result[0]["value"][1])

    except Exception:
        return None


def get_acceptance_rate():
    query = """
    rate(vllm:spec_decode_num_accepted_tokens_total[1m])
    /
    rate(vllm:spec_decode_num_draft_tokens_total[1m])
    """
    value = query_prometheus(query)
    return value * 100 if value is not None else None


run = st.button("Run baseline + speculative concurrently", type="primary")

st.divider()

# Metrics are always visible near the top.
st.subheader("Live comparison metrics")

top1, top2, top3, top4 = st.columns(4)
baseline_latency_slot = top1.empty()
spec_latency_slot = top2.empty()
speedup_slot = top3.empty()
acceptance_slot = top4.empty()

bottom1, bottom2, bottom3, bottom4 = st.columns(4)
baseline_ttft_slot = bottom1.empty()
spec_ttft_slot = bottom2.empty()
baseline_tps_slot = bottom3.empty()
spec_tps_slot = bottom4.empty()

baseline_latency_slot.metric("Baseline latency", "—")
spec_latency_slot.metric("Spec latency", "—")
speedup_slot.metric("Speedup", "—")
acceptance_slot.metric("Draft acceptance rate", "—")

baseline_ttft_slot.metric("Baseline TTFT", "—")
spec_ttft_slot.metric("Spec TTFT", "—")
baseline_tps_slot.metric("Baseline chunks/sec", "—")
spec_tps_slot.metric("Spec chunks/sec", "—")

st.divider()

# Outputs are below the metrics, side by side.
output_col1, output_col2 = st.columns(2)

with output_col1:
    st.subheader("Baseline: target model only")
    baseline_box = st.empty()

with output_col2:
    st.subheader("Speculative: target + draft model")
    spec_box = st.empty()


if run:
    q = queue.Queue()

    baseline_thread = threading.Thread(
        target=stream_request,
        args=("baseline", baseline_client, TARGET_MODEL, prompt, q),
    )

    spec_thread = threading.Thread(
        target=stream_request,
        args=("speculative", spec_client, TARGET_MODEL, prompt, q),
    )

    baseline_text = ""
    spec_text = ""

    baseline_result = None
    spec_result = None

    baseline_box.markdown("Running baseline...")
    spec_box.markdown("Running speculative...")

    baseline_thread.start()
    spec_thread.start()

    while baseline_thread.is_alive() or spec_thread.is_alive() or not q.empty():
        try:
            name, event_type, payload = q.get(timeout=0.1)

            if event_type == "token":
                if name == "baseline":
                    baseline_text += payload
                    baseline_box.markdown(baseline_text)
                else:
                    spec_text += payload
                    spec_box.markdown(spec_text)

            elif event_type == "ttft":
                if name == "baseline":
                    baseline_ttft_slot.metric("Baseline TTFT", f"{payload:.2f}s")
                else:
                    spec_ttft_slot.metric("Spec TTFT", f"{payload:.2f}s")

            elif event_type == "done":
                if name == "baseline":
                    baseline_result = payload
                    baseline_latency_slot.metric("Baseline latency", f"{payload['latency']:.2f}s")
                    baseline_tps_slot.metric("Baseline chunks/sec", f"{payload['chunks_per_second']:.1f}")
                else:
                    spec_result = payload
                    spec_latency_slot.metric("Spec latency", f"{payload['latency']:.2f}s")
                    spec_tps_slot.metric("Spec chunks/sec", f"{payload['chunks_per_second']:.1f}")

            elif event_type == "error":
                if name == "baseline":
                    baseline_box.error(payload)
                else:
                    spec_box.error(payload)

        except queue.Empty:
            pass

    acceptance_rate = get_acceptance_rate()

    if acceptance_rate is not None:
        acceptance_slot.metric("Draft acceptance rate", f"{acceptance_rate:.1f}%")
    else:
        acceptance_slot.metric("Draft acceptance rate", "N/A")

    if baseline_result and spec_result:
        speedup = baseline_result["latency"] / spec_result["latency"]
        speedup_slot.metric("Speedup", f"{speedup:.2f}x")