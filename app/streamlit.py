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


DEMO1_PROMPTS = {
    "Structured / JSON": """Generate a JSON array of 20 synthetic API request logs.

Each object must follow this schema:
{
  "request_id": string,
  "endpoint": string,
  "method": "GET" | "POST" | "DELETE",
  "status_code": number,
  "latency_ms": number,
  "cached": boolean
}

Return only valid JSON.""",

    "Code / technical": """Write a Python function that batches incoming inference requests by arrival time.

Requirements:
- Include type hints.
- Include comments.
- Handle empty input.
- Include a short example usage.""",

    "Creative / open-ended": """Write a short fictional story about an engineer debugging a mysterious latency spike before a live demo."""
}


def make_context(target_tokens: int) -> str:
    # Rough estimate: 1 token ≈ 4 characters.
    target_chars = target_tokens * 4
    filler = (
        "Background context: vLLM serves LLM requests using batching, KV cache, "
        "prefill, decode, CUDA kernels, scheduling, and memory management. "
        "Speculative decoding uses a smaller draft model to propose tokens that "
        "the target model verifies. "
    )
    return (filler * ((target_chars // len(filler)) + 1))[:target_chars]


def query_prometheus(query: str):
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=3,
        )
        response.raise_for_status()
        result = response.json()["data"]["result"]

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


def run_streaming_request(client, prompt: str, max_tokens: int, temperature: float, output_slot):
    start = time.perf_counter()
    first_token_time = None
    text = ""
    chunks = 0

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
            chunks += 1
            output_slot.markdown(text)

    total_latency = time.perf_counter() - start

    return {
        "latency": total_latency,
        "ttft": first_token_time,
        "chunks": chunks,
        "chars": len(text),
        "chars_per_second": len(text) / total_latency if total_latency > 0 else 0,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, text


st.set_page_config(page_title="vLLM Speculative Decoding Demo", layout="wide")
st.title("vLLM Baseline vs Speculative Decoding")

st.sidebar.header("Demo controls")

demo = st.sidebar.radio(
    "Demo",
    [
        "Demo 1: Workload type",
        "Demo 2: Context length",
        "Demo 3: Production stress note",
    ],
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

if demo == "Demo 1: Workload type":
    workload = st.sidebar.radio("Workload type", list(DEMO1_PROMPTS.keys()))
    prompt = DEMO1_PROMPTS[workload]
    max_tokens = st.sidebar.slider("Max generated tokens", 128, 1024, 512, step=64)

elif demo == "Demo 2: Context length":
    context_choice = st.sidebar.radio("Prompt/context size", ["8K", "24K", "32K"])

    context_tokens = {
        "8K": 7500,
        "24K": 23500,
        "32K": 31500,
    }[context_choice]

    max_tokens = 512

    task = (
        "Using only the context above, summarize the most important debugging steps "
        "for investigating vLLM latency in 5 concise bullet points."
    )

    prompt = make_context(context_tokens) + "\n\n" + task

else:
    st.info(
        "Demo 3 should be driven by your load generator, not manual Streamlit clicks. "
        "Run concurrency = 1, 8, 16, and watch Grafana for p50/p95 latency, "
        "GPU utilization, queue depth, running requests, tokens/sec, and acceptance rate."
    )
    st.stop()


with st.expander("Prompt preview", expanded=False):
    st.write(f"Approx prompt characters: {len(prompt):,}")
    st.text_area("Prompt", prompt, height=260)


if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None

if "spec_result" not in st.session_state:
    st.session_state.spec_result = None

if "baseline_output" not in st.session_state:
    st.session_state.baseline_output = ""

if "spec_output" not in st.session_state:
    st.session_state.spec_output = ""


control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    run_baseline = st.button("Run Baseline", type="primary")

with control_col2:
    run_spec = st.button("Run Speculative", type="secondary")

with control_col3:
    clear = st.button("Clear Results")

if clear:
    st.session_state.baseline_result = None
    st.session_state.spec_result = None
    st.session_state.baseline_output = ""
    st.session_state.spec_output = ""
    st.rerun()


st.divider()

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.subheader("Baseline")
    b1, b2, b3 = st.columns(3)
    baseline_latency_slot = b1.empty()
    baseline_ttft_slot = b2.empty()
    baseline_rate_slot = b3.empty()

with metric_col2:
    st.subheader("Speculative")
    s1, s2, s3 = st.columns(3)
    spec_latency_slot = s1.empty()
    spec_ttft_slot = s2.empty()
    spec_rate_slot = s3.empty()

    s4, s5 = st.columns(2)
    speedup_slot = s4.empty()
    acceptance_slot = s5.empty()


baseline_latency_slot.metric("Total latency", "—")
baseline_ttft_slot.metric("TTFT", "—")
baseline_rate_slot.metric("Chars/sec", "—")

spec_latency_slot.metric("Total latency", "—")
spec_ttft_slot.metric("TTFT", "—")
spec_rate_slot.metric("Chars/sec", "—")
speedup_slot.metric("Speedup", "—")
acceptance_slot.metric("Draft acceptance", "—")

verdict_slot = st.empty()

st.divider()

output_col1, output_col2 = st.columns(2)

with output_col1:
    st.subheader("Baseline output")
    baseline_output_slot = st.empty()

with output_col2:
    st.subheader("Speculative output")
    spec_output_slot = st.empty()


def render_results():
    baseline_result = st.session_state.baseline_result
    spec_result = st.session_state.spec_result

    if baseline_result:
        baseline_latency_slot.metric("Total latency", f"{baseline_result['latency']:.2f}s")
        baseline_ttft_slot.metric(
            "TTFT",
            f"{baseline_result['ttft']:.2f}s" if baseline_result["ttft"] else "N/A",
        )
        baseline_rate_slot.metric("Chars/sec", f"{baseline_result['chars_per_second']:.0f}")
        baseline_output_slot.markdown(st.session_state.baseline_output)

    if spec_result:
        spec_latency_slot.metric("Total latency", f"{spec_result['latency']:.2f}s")
        spec_ttft_slot.metric(
            "TTFT",
            f"{spec_result['ttft']:.2f}s" if spec_result["ttft"] else "N/A",
        )
        spec_rate_slot.metric("Chars/sec", f"{spec_result['chars_per_second']:.0f}")

        acceptance = spec_result.get("acceptance_rate")
        acceptance_slot.metric(
            "Draft acceptance",
            f"{acceptance:.1f}%" if acceptance is not None else "N/A",
        )

        spec_output_slot.markdown(st.session_state.spec_output)

    if baseline_result and spec_result:
        speedup = baseline_result["latency"] / spec_result["latency"]
        speedup_slot.metric("Speedup", f"{speedup:.2f}x")

        delta = baseline_result["latency"] - spec_result["latency"]
        pct = (delta / baseline_result["latency"]) * 100

        verdict_slot.info(
            f"Speculative decoding changed total latency by {pct:.1f}% "
            f"({delta:.2f}s difference)."
        )


if run_baseline:
    baseline_output_slot.markdown("Running baseline...")
    try:
        result, text = run_streaming_request(
            baseline_client,
            prompt,
            max_tokens,
            temperature,
            baseline_output_slot,
        )
        st.session_state.baseline_result = result
        st.session_state.baseline_output = text
        st.rerun()
    except Exception as e:
        st.error(f"Baseline request failed: {e}")


if run_spec:
    spec_output_slot.markdown("Running speculative...")
    try:
        result, text = run_streaming_request(
            spec_client,
            prompt,
            max_tokens,
            temperature,
            spec_output_slot,
        )
        result["acceptance_rate"] = get_acceptance_rate()
        st.session_state.spec_result = result
        st.session_state.spec_output = text
        st.rerun()
    except Exception as e:
        st.error(f"Speculative request failed: {e}")


render_results()