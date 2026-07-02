import os
import time
import requests
import threading
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
# Fixed internal framework imports
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

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

def get_detailed_spec_metrics():
    accepted_query = "sum(vllm:spec_decode_num_accepted_tokens_total)"
    drafted_query = "sum(vllm:spec_decode_num_draft_tokens_total)"
    
    accepted = query_prometheus(accepted_query)
    drafted = query_prometheus(drafted_query)
    
    if accepted and drafted:
        rate = (accepted / drafted) * 100
        return int(accepted), int(drafted), rate
    return None, None, None

def stream_engine(client, prompt: str, max_tokens: int, temp: float, output_slot):
    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0

    try:
        stream = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                text += delta
                token_count += 1
                output_slot.markdown(text)

        total_latency = time.perf_counter() - start
        return {
            "latency": total_latency,
            "ttft": first_token_time,
            "tokens": token_count,
            "tokens_per_second": token_count / total_latency if total_latency > 0 else 0,
            "output_text": text
        }
    except Exception as e:
        output_slot.error(f"Execution failed: {e}")
        return None

st.set_page_config(page_title="vLLM Speculative Decoding Demo", layout="wide")
st.title("⚡ vLLM Live Inference Optimization Arena")

st.sidebar.header("Execution Profile")
demo = st.sidebar.radio(
    "Select Scenario",
    ["Demo 1: Workload Type Predictability", "Demo 2: Extreme Context Scaling", "Demo 3: High-Concurrency Stress Test"]
)

temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, step=0.1)

if demo == "Demo 1: Workload Type Predictability":
    workload = st.sidebar.radio("Workload Constraints", list(DEMO1_PROMPTS.keys()))
    prompt = DEMO1_PROMPTS[workload]
    max_tokens = st.sidebar.slider("Max Output Length", 128, 1024, 384, step=64)

elif demo == "Demo 2: Extreme Context Scaling":
    context_choice = st.sidebar.radio("Prompt/Context Size", ["8K", "24K", "32K"])
    context_tokens = {"8K": 7500, "24K": 23500, "32K": 31500}[context_choice]
    max_tokens = 256
    task = "Using only the background context above, summarize the architecture of vLLM in 3 crisp sentences."
    prompt = make_context(context_tokens) + "\n\nTask: " + task

else:
    st.info("💡 **Demo 3 Instructions:** Hit your server backend using your external load generator framework. Monitor system-wide latency variance live inside Grafana.")
    st.stop()

with st.expander("Inspect Raw Input Payload", expanded=False):
    st.text_area("Compiled Prompt System Vector", prompt, height=200)

st.divider()

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    execute_race = st.button("🚀 Run Simultaneous Race", type="primary")
with col_btn2:
    if st.button("Clear Playback Area"):
        st.session_state.clear()
        st.rerun()

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.subheader("🤖 Traditional Target Engine (Baseline)")
    b_latency = st.empty()
    b_ttft = st.empty()
    b_tps = st.empty()
    st.caption("Generation Sandbox")
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Acceleration Layer")
    s_latency = st.empty()
    s_ttft = st.empty()
    s_tps = st.empty()
    st.caption("Generation Sandbox")
    spec_output_slot = st.empty()

st.divider()
st.subheader("📊 Architectural Evaluation Summary")
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    speedup_stat_slot = st.empty()
with summary_col2:
    spec_breakdown_slot = st.empty()

b_latency.metric("Total Generation Time", "—")
b_ttft.metric("Time to First Token (Prefill)", "—")
b_tps.metric("Generation Throughput", "—")
s_latency.metric("Total Generation Time", "—")
s_ttft.metric("Time to First Token (Prefill)", "—")
s_tps.metric("Generation Throughput", "—")
speedup_stat_slot.metric("Net Speedup Delta", "—")
spec_breakdown_slot.metric("Draft Acceptance Metric", "—")

if execute_race:
    baseline_output_slot.info("Awaiting Stream Dispatch...")
    spec_output_slot.info("Awaiting Stream Dispatch...")

    # Capture the thread context cleanly using correct names
    ctx = get_script_run_ctx()

    def run_with_runtime_context(client, prompt, max_tokens, temp, slot):
        # Bind parent script run context explicitly to this specific thread execution block
        add_script_run_ctx(threading.current_thread(), ctx)
        return stream_engine(client, prompt, max_tokens, temp, slot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_baseline = executor.submit(run_with_runtime_context, baseline_client, prompt, max_tokens, temperature, baseline_output_slot)
        future_spec = executor.submit(run_with_runtime_context, spec_client, prompt, max_tokens, temperature, spec_output_slot)
        
        while not (future_baseline.done() and future_spec.done()):
            time.sleep(0.1)

    base_res = future_baseline.result()
    spec_res = future_spec.result()

    if base_res:
        b_latency.metric("Total Generation Time", f"{base_res['latency']:.2f}s")
        b_ttft.metric("Time to First Token (Prefill)", f"{base_res['ttft']:.3f}s")
        b_tps.metric("Generation Throughput", f"{base_res['tokens_per_second']:.1f} tok/s")

    if spec_res:
        s_latency.metric("Total Generation Time", f"{spec_res['latency']:.2f}s")
        s_ttft.metric("Time to First Token (Prefill)", f"{spec_res['ttft']:.3f}s")
        s_tps.metric("Generation Throughput", f"{spec_res['tokens_per_second']:.1f} tok/s")
        
        accepted, drafted, rate = get_detailed_spec_metrics()
        if rate:
            spec_breakdown_slot.metric(
                "Draft Token Acceptance", 
                f"{rate:.1f}%", 
                help=f"System Breakdown: Total Draft Proposed Tokens: {drafted} | Total System Verified and Accepted Tokens: {accepted}"
            )

    if base_res and spec_res:
        net_speedup = base_res['latency'] / spec_res['latency']
        speedup_stat_slot.metric("Net Speedup Delta", f"{net_speedup:.2f}x Ratio")