import os
import time
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

if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None

if "spec_result" not in st.session_state:
    st.session_state.spec_result = None

if "baseline_output" not in st.session_state:
    st.session_state.baseline_output = ""

if "spec_output" not in st.session_state:
    st.session_state.spec_output = ""


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


def run_streaming_request(mode_name, client):
    start = time.perf_counter()
    first_token_time = None
    text = ""
    chunk_count = 0

    output_placeholder = (
        baseline_output_slot if mode_name == "baseline" else spec_output_slot
    )

    try:
        stream = client.chat.completions.create(
            model=TARGET_MODEL,
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

                text += delta
                chunk_count += 1
                output_placeholder.markdown(text)

        total_latency = time.perf_counter() - start

        result = {
            "latency": total_latency,
            "ttft": first_token_time,
            "chunks": chunk_count,
            "chunks_per_second": chunk_count / total_latency if total_latency > 0 else 0,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt": prompt,
        }

        return result, text

    except Exception as e:
        st.error(f"{mode_name} request failed: {e}")
        return None, text


def render_metrics():
    baseline_result = st.session_state.baseline_result
    spec_result = st.session_state.spec_result

    if baseline_result:
        baseline_latency_slot.metric(
            "Total latency",
            f"{baseline_result['latency']:.2f}s",
        )
        baseline_ttft_slot.metric(
            "Time to first token",
            f"{baseline_result['ttft']:.2f}s"
            if baseline_result["ttft"] is not None
            else "N/A",
        )
        baseline_tps_slot.metric(
            "Chunks/sec",
            f"{baseline_result['chunks_per_second']:.1f}",
        )
        baseline_output_slot.markdown(st.session_state.baseline_output)

    if spec_result:
        spec_latency_slot.metric(
            "Total latency",
            f"{spec_result['latency']:.2f}s",
        )
        spec_ttft_slot.metric(
            "Time to first token",
            f"{spec_result['ttft']:.2f}s"
            if spec_result["ttft"] is not None
            else "N/A",
        )
        spec_tps_slot.metric(
            "Chunks/sec",
            f"{spec_result['chunks_per_second']:.1f}",
        )

        acceptance = spec_result.get("acceptance_rate")
        acceptance_slot.metric(
            "Draft acceptance rate",
            f"{acceptance:.1f}%" if acceptance is not None else "N/A",
        )

        spec_output_slot.markdown(st.session_state.spec_output)

    if baseline_result and spec_result:
        speedup = baseline_result["latency"] / spec_result["latency"]
        speedup_slot.metric("Speedup", f"{speedup:.2f}x")

        latency_delta = baseline_result["latency"] - spec_result["latency"]
        latency_pct = (latency_delta / baseline_result["latency"]) * 100

        verdict_slot.info(
            f"Speculative decoding changed total latency by {latency_pct:.1f}% "
            f"({latency_delta:.2f}s difference)."
        )


st.divider()

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    run_baseline = st.button("Run Baseline", type="primary")

with control_col2:
    run_spec = st.button("Run Speculative", type="secondary")

with control_col3:
    clear_results = st.button("Clear Saved Results")

if clear_results:
    st.session_state.baseline_result = None
    st.session_state.spec_result = None
    st.session_state.baseline_output = ""
    st.session_state.spec_output = ""
    st.rerun()

st.divider()

metrics_left, metrics_right = st.columns(2)

with metrics_left:
    st.subheader("Baseline")

    b1, b2, b3 = st.columns(3)
    baseline_latency_slot = b1.empty()
    baseline_ttft_slot = b2.empty()
    baseline_tps_slot = b3.empty()

    baseline_latency_slot.metric("Total latency", "—")
    baseline_ttft_slot.metric("Time to first token", "—")
    baseline_tps_slot.metric("Chunks/sec", "—")

with metrics_right:
    st.subheader("Speculative")

    s1, s2, s3 = st.columns(3)
    spec_latency_slot = s1.empty()
    spec_ttft_slot = s2.empty()
    spec_tps_slot = s3.empty()

    s4, s5 = st.columns(2)
    speedup_slot = s4.empty()
    acceptance_slot = s5.empty()

    spec_latency_slot.metric("Total latency", "—")
    spec_ttft_slot.metric("Time to first token", "—")
    spec_tps_slot.metric("Chunks/sec", "—")
    speedup_slot.metric("Speedup", "—")
    acceptance_slot.metric("Draft acceptance rate", "—")

st.divider()

verdict_slot = st.empty()

output_col1, output_col2 = st.columns(2)

with output_col1:
    st.subheader("Baseline output")
    baseline_output_slot = st.empty()

with output_col2:
    st.subheader("Speculative output")
    spec_output_slot = st.empty()


if run_baseline:
    baseline_output_slot.markdown("Running baseline...")

    result, text = run_streaming_request("baseline", baseline_client)

    if result:
        st.session_state.baseline_result = result
        st.session_state.baseline_output = text
        st.rerun()


if run_spec:
    spec_output_slot.markdown("Running speculative...")

    result, text = run_streaming_request("speculative", spec_client)

    if result:
        result["acceptance_rate"] = get_acceptance_rate()
        st.session_state.spec_result = result
        st.session_state.spec_output = text
        st.rerun()


render_metrics()