import os
import time
import requests
import threading
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

BASELINE_URL = "http://127.0.0.1:8000/v1"
SPEC_URL = "http://127.0.0.1:8001/v1"
PROMETHEUS_URL = "http://127.0.0.1:9090"

TARGET_MODEL = os.getenv("TARGET_MODEL")

baseline_client = OpenAI(base_url=BASELINE_URL, api_key="EMPTY")
spec_client = OpenAI(base_url=SPEC_URL, api_key="EMPTY")

JSON_PROMPT = """Generate a JSON array of 20 synthetic API request logs.
Each object must follow this schema:
{
  "request_id": string,
  "endpoint": string,
  "method": "GET" | "POST" | "DELETE",
  "status_code": number,
  "latency_ms": number,
  "cached": boolean
}
Return only valid JSON."""

CREATIVE_PROMPT = """Write a short fictional story about an engineer debugging a mysterious latency spike before a live demo."""

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
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else None
    except Exception:
        return None

def get_detailed_spec_metrics():
    accepted = query_prometheus("sum(vllm:spec_decode_num_accepted_tokens_total)")
    drafted = query_prometheus("sum(vllm:spec_decode_num_draft_tokens_total)")
    if accepted and drafted:
        return int(accepted), int(drafted), (accepted / drafted) * 100
    return None, None, None

def stream_engine(client, prompt: str, max_tokens: int, temp: float, output_slot):
    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0
    try:
        stream = client.chat.completions.create(
            model=TARGET_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temp, stream=True,
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
            "latency": total_latency, "ttft": first_token_time, "tokens": token_count,
            "tokens_per_second": token_count / total_latency if total_latency > 0 else 0
        }
    except Exception as e:
        output_slot.error(f"Execution failed: {e}")
        return None

st.set_page_config(page_title="vLLM Speculative Decoding Arena", layout="wide")
st.title("⚡ vLLM Live Inference Optimization Arena")

st.sidebar.header("Navigation")
demo = st.sidebar.radio("Select Demo Scenario", ["Demo 1: Workload Predictability", "Demo 2: Context Scaling", "Demo 3: Production Stress Note"])
temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, step=0.1)

# Configure Prompts based on Selection
if demo == "Demo 1: Workload Predictability":
    tab1, tab2 = st.tabs(["📋 Structured Task (JSON)", "🎨 Open-Ended Task (Creative)"])
    with tab1:
        st.caption("Testing high predictability infrastructure pathways.")
    with tab2:
        st.caption("Testing high variance vocabulary spaces.")
    
    # Simple check to see which tab layout is active to select prompt
    workload_type = "JSON" if tab1 else "Creative" 
    max_tokens = st.sidebar.slider("Max Output Tokens", 128, 1024, 384, step=64)
    prompt = JSON_PROMPT if workload_type == "JSON" else CREATIVE_PROMPT

elif demo == "Demo 2: Context Scaling":
    context_choice = st.sidebar.radio("Prompt Context Window Size", ["8K", "24K", "32K"])
    context_tokens = {"8K": 7500, "24K": 23500, "32K": 31500}[context_choice]
    max_tokens = 256
    prompt = make_context(context_tokens) + "\n\nTask: Summarize the architecture of vLLM in 3 concise sentences."

else:
    st.info("💡 **Demo 3 Instructions:** Transition to your live Grafana dashboard layout now. Fire up your external load generator to showcase concurrent scaling overhead.")
    st.stop()

st.divider()

# Core Execution Interface
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    execute_race = st.button("🚀 Run Simultaneous Race", type="primary")
with col_btn2:
    if st.button("Clear Playback Matrix"):
        st.session_state.clear()
        st.rerun()

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.subheader("🤖 Traditional Baseline Engine")
    b_latency = st.empty()
    b_ttft = st.empty()
    b_tps = st.empty()
    st.caption("Streaming Workspace")
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    s_latency = st.empty()
    s_ttft = st.empty()
    s_tps = st.empty()
    st.caption("Streaming Workspace")
    spec_output_slot = st.empty()

st.divider()
st.subheader("📊 Performance Evaluation Insights")
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    speedup_stat_slot = st.empty()
with summary_col2:
    spec_breakdown_slot = st.empty()

# Default Placeholders
b_latency.metric("Total Request Time", "—")
b_ttft.metric("Time to First Token (TTFT)", "—")
b_tps.metric("Throughput Speed", "—")
s_latency.metric("Total Request Time", "—")
s_ttft.metric("Time to First Token (TTFT)", "—")
s_tps.metric("Throughput Speed", "—")
speedup_stat_slot.metric("Net Speedup Multiplier", "—")
spec_breakdown_slot.metric("Draft Acceptance Rate", "—")

if execute_race:
    baseline_output_slot.info("Warming baseline streaming pipes...")
    spec_output_slot.info("Warming speculative acceleration layer...")

    ctx = get_script_run_ctx()

    def run_with_runtime_context(client, prompt, max_tokens, temp, slot):
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
        b_latency.metric("Total Request Time", f"{base_res['latency']:.2f}s")
        b_ttft.metric("Time to First Token (TTFT)", f"{base_res['ttft']:.3f}s")
        b_tps.metric("Throughput Speed", f"{base_res['tokens_per_second']:.1f} tok/s")

    if spec_res:
        s_latency.metric("Total Request Time", f"{spec_res['latency']:.2f}s")
        s_ttft.metric("Time to First Token (TTFT)", f"{spec_res['ttft']:.3f}s")
        s_tps.metric("Throughput Speed", f"{spec_res['tokens_per_second']:.1f} tok/s")
        
        accepted, drafted, rate = get_detailed_spec_metrics()
        if rate:
            spec_breakdown_slot.metric(
                "Draft Acceptance Rate", f"{rate:.1f}%",
                help=f"Telemetry: Accepted {accepted} tokens out of {drafted} proposed guesses."
            )

    if base_res and spec_res:
        net_speedup = base_res['latency'] / spec_res['latency']
        speedup_stat_slot.metric("Net Speedup Multiplier", f"{net_speedup:.2f}x Faster")