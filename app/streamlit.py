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

if "json_results" not in st.session_state:
    st.session_state.json_results = {"base_res": None, "spec_res": None}
if "creative_results" not in st.session_state:
    st.session_state.creative_results = {"base_res": None, "spec_res": None}

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
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0

def stream_engine(client, prompt: str, max_tokens: int, temp: float, output_slot):
    start_accepted = query_prometheus("sum(vllm:spec_decode_num_accepted_tokens_total)")
    start_drafted = query_prometheus("sum(vllm:spec_decode_num_draft_tokens_total)")

    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=TARGET_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temp, stream=True,
            stream_options={"include_usage": True}
        )
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    text += delta
                    output_slot.markdown(text)
            
            if hasattr(chunk, "usage") and chunk.usage is not None:
                token_count = chunk.usage.completion_tokens

        total_latency = time.perf_counter() - start
        generation_time = total_latency - (first_token_time or 0)
        
        if token_count == 0:
            token_count = len(text.split())

        tpot = (generation_time / token_count) * 1000 if token_count > 0 else 0
        tps = token_count / total_latency if total_latency > 0 else 0
        
        time.sleep(0.4)
        end_accepted = query_prometheus("sum(vllm:spec_decode_num_accepted_tokens_total)")
        end_drafted = query_prometheus("sum(vllm:spec_decode_num_draft_tokens_total)")
        
        run_accepted = end_accepted - start_accepted
        run_drafted = end_drafted - start_drafted
        run_rate = (run_accepted / run_drafted) * 100 if run_drafted > 0 else 0.0

        return {
            "latency": total_latency, "ttft": first_token_time, "tokens": token_count,
            "tokens_per_second": tps, "tpot": tpot, "text": text, "run_rate": run_rate
        }
    except Exception as e:
        output_slot.error(f"Execution failed: {e}")
        return None

st.set_page_config(page_title="vLLM Speculative Decoding Arena", layout="wide")
st.title("⚡ vLLM Live Inference Optimization Arena")

st.sidebar.header("Navigation")
demo = st.sidebar.radio("Select Demo Scenario", ["Demo 1: Workload Predictability", "Demo 2: Context Scaling", "Demo 3: Production Stress Note"])
temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, step=0.1)

if demo == "Demo 1: Workload Predictability":
    tab_selection = st.radio("Workload Pathway", ["📋 Structured Task (JSON)", "🎨 Open-Ended Task (Creative)"], horizontal=True)
    max_tokens = st.sidebar.slider("Max Output Tokens", 128, 1024, 384, step=64)
    
    if tab_selection == "📋 Structured Task (JSON)":
        active_key = "json_results"
        prompt = JSON_PROMPT
    else:
        active_key = "creative_results"
        prompt = CREATIVE_PROMPT
        
    with st.expander("🔍 View Prompt Running on Stage", expanded=True):
        st.code(prompt, language="text")

elif demo == "Demo 2: Context Scaling":
    active_key = None
    context_choice = st.sidebar.radio("Prompt Context Window Size", ["8K", "24K", "32K"])
    context_tokens = {"8K": 7500, "24K": 23500, "32K": 31500}[context_choice]
    max_tokens = 256
    raw_context = make_context(context_tokens)
    instruction = "\n\nTask: Summarize the architecture of vLLM in 3 concise sentences."
    prompt = raw_context + instruction
    
    with st.expander("🔍 View Prompt Running on Stage (Large Context)", expanded=True):
        st.write(f"**Prefix Padding Length:** ~{context_tokens} tokens")
        st.code(instruction.strip(), language="text")

else:
    st.info("💡 **Demo 3 Instructions:** Transition to your live Grafana dashboard layout now. Fire up your external load generator to showcase concurrent scaling overhead.")
    st.stop()

st.divider()

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
    b_tpot = st.empty()
    speedup_slot = st.empty()
    st.caption("Streaming Workspace")
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    s_latency = st.empty()
    s_ttft = st.empty()
    s_tps = st.empty()
    s_tpot = st.empty()
    spec_breakdown_slot = st.empty()
    st.caption("Streaming Workspace")
    spec_output_slot = st.empty()

def display_persisted_metrics(base_res, spec_res):
    if base_res:
        b_latency.metric("Total Request Time", f"{base_res['latency']:.2f}s")
        b_ttft.metric("Time to First Token (TTFT)", f"{base_res['ttft']:.3f}s")
        b_tps.metric("Output Tokens Per Second", f"{base_res['tokens_per_second']:.1f} tok/s")
        b_tpot.metric("Time Per Output Token (TPOT)", f"{base_res['tpot']:.1f} ms/tok")
        baseline_output_slot.markdown(base_res['text'])
    else:
        b_latency.metric("Total Request Time", "—")
        b_ttft.metric("Time to First Token (TTFT)", "—")
        b_tps.metric("Output Tokens Per Second", "—")
        b_tpot.metric("Time Per Output Token (TPOT)", "—")

    if spec_res:
        s_latency.metric("Total Request Time", f"{spec_res['latency']:.2f}s")
        s_ttft.metric("Time to First Token (TTFT)", f"{spec_res['ttft']:.3f}s")
        s_tps.metric("Output Tokens Per Second", f"{spec_res['tokens_per_second']:.1f} tok/s")
        s_tpot.metric("Time Per Output Token (TPOT)", f"{spec_res['tpot']:.1f} ms/tok")
        spec_output_slot.markdown(spec_res['text'])
        spec_breakdown_slot.metric("Draft Acceptance Rate (This Run)", f"{spec_res['run_rate']:.1f}%")
    else:
        s_latency.metric("Total Request Time", "—")
        s_ttft.metric("Time to First Token (TTFT)", "—")
        s_tps.metric("Output Tokens Per Second", "—")
        s_tpot.metric("Time Per Output Token (TPOT)", "—")
        spec_breakdown_slot.metric("Draft Acceptance Rate (This Run)", "—")

    if base_res and spec_res and spec_res['latency'] > 0:
        net_speedup = base_res['latency'] / spec_res['latency']
        speedup_slot.metric("Net Speedup Performance Delta", f"{net_speedup:.2f}x Faster")
    else:
        speedup_slot.metric("Net Speedup Performance Delta", "—")

if demo == "Demo 1: Workload Predictability" and active_key:
    display_persisted_metrics(
        st.session_state[active_key]["base_res"],
        st.session_state[active_key]["spec_res"]
    )
else:
    display_persisted_metrics(None, None)

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

    if demo == "Demo 1: Workload Predictability" and active_key:
        st.session_state[active_key]["base_res"] = base_res
        st.session_state[active_key]["spec_res"] = spec_res
        st.rerun()
    else:
        display_persisted_metrics(base_res, spec_res)