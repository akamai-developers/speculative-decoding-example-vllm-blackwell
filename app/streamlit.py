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

# ==============================================================================
# PROMPT DEFINITIONS
# ==============================================================================

# Demo 1 Prompts
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

# Demo 2 Enterprise Compliance Prompts
COMPLIANCE_TEXT_BLOCK = """
[TIMESTAMP 00:14:22] AUDIT RECORD EXCERPT: The entity's primary fiscal steering committee reviewed unhedged international volatile asset surfaces. Operational capital preserves were systematically re-allocated across regional holdings to buffer liquidity degradation vectors. The CFO affirmed during the live presentation that while foreign exchange metrics show persistent downward pressure, our current exposure matrix maintains a highly predictable, risk-adjusted profile without relying on artificial valuation multipliers. Moving on to secondary market derivatives...
"""

INSTRUCTION_BLOCK = """
\n\n[REGULATORY AUDIT MANDATE]
Analyze the entire transaction audit logging history above. 
Verify if the financial executor explicitly guaranteed yields or used the phrase 'guaranteed yield' regarding volatile asset classes.
Output exactly one concise sentence declaring your final compliance determination and cite the section timestamp.
"""

D2_PROMPTS = {
    "8K Context": (COMPLIANCE_TEXT_BLOCK * 70) + INSTRUCTION_BLOCK,
    "24K Context": (COMPLIANCE_TEXT_BLOCK * 210) + INSTRUCTION_BLOCK,
    "32K Context": (COMPLIANCE_TEXT_BLOCK * 280) + INSTRUCTION_BLOCK
}

# ==============================================================================
# SESSION STATE INITIALIZATION (PERSISTENCE)
# ==============================================================================

if "d1_state" not in st.session_state:
    st.session_state.d1_state = {
        "json_results": {"base_res": None, "spec_res": None},
        "creative_results": {"base_res": None, "spec_res": None}
    }

if "d2_state" not in st.session_state:
    st.session_state.d2_state = {
        "8K Context": {"base_res": None, "spec_res": None},
        "24K Context": {"base_res": None, "spec_res": None},
        "32K Context": {"base_res": None, "spec_res": None}
    }

# ==============================================================================
# ENGINE CORE FUNCTIONALITY
# ==============================================================================

def query_prometheus(query: str):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0

def stream_engine(client, prompt: str, max_tokens: int, temp: float, output_slot):
    # Snapshot delta baseline counters for acceptance tracking (Only evaluated/needed for Demo 1)
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
        output_slot.error(f"Inference Fault: {e}")
        return None

# ==============================================================================
# UI SETUP & ROUTING
# ==============================================================================

st.set_page_config(page_title="vLLM Inference Optimization Arena", layout="wide")
st.title("⚡ vLLM Live Inference Optimization Arena")

st.sidebar.header("Navigation")
demo = st.sidebar.radio("Select Demo Scenario", ["Demo 1: Workload Predictability", "Demo 2: Context Scaling", "Demo 3: Production Stress Note"])

# Route parameters based on scenario selection
if demo == "Demo 1: Workload Predictability":
    temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, step=0.1)
    max_tokens = st.sidebar.slider("Max Output Tokens", 128, 1024, 384, step=64)
    
    tab_selection = st.radio("Workload Pathway", ["📋 Structured Task (JSON)", "🎨 Open-Ended Task (Creative)"], horizontal=True)
    if tab_selection == "📋 Structured Task (JSON)":
        active_key = "json_results"
        prompt = JSON_PROMPT
    else:
        active_key = "creative_results"
        prompt = CREATIVE_PROMPT
        
    with st.expander("🔍 View Prompt Running on Stage", expanded=True):
        st.code(prompt, language="text")

elif demo == "Demo 2: Context Scaling":
    temperature = 0.0  # Force deterministic extraction for audit
    max_tokens = 128   # Short compliance verdict response
    
    context_tier = st.radio("Select Active Context Window Size", ["8K Context", "24K Context", "32K Context"], horizontal=True)
    prompt = D2_PROMPTS[context_tier]
    
    with st.expander("🔍 View Active Compliance Prompt Footprint Running on GPU", expanded=True):
        st.write(f"**Total Document Payload Footprint:** ~{context_tier}")
        st.text(prompt[:1200] + "\n\n [... MIDDLE OMITTED FOR VIEWPORT ...]\n\n" + INSTRUCTION_BLOCK)

else:
    st.info("💡 **Demo 3 Instructions:** Transition to your live Grafana dashboard layout now. Fire up your external load generator to showcase concurrent scaling overhead.")
    st.stop()

st.divider()

# Action Buttons
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    execute_race = st.button("🚀 Run Simultaneous Race", type="primary")
with col_btn2:
    if st.button("Clear Playback Matrices"):
        if demo == "Demo 1: Workload Predictability":
            st.session_state.d1_state = {"json_results": {"base_res": None, "spec_res": None}, "creative_results": {"base_res": None, "spec_res": None}}
        else:
            st.session_state.d2_state = {k: {"base_res": None, "spec_res": None} for k in D2_PROMPTS.keys()}
        st.rerun()

# Layout Metric Columns
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.subheader("🤖 Traditional Baseline Engine")
    b_latency, b_ttft, b_tps, b_tpot = st.empty(), st.empty(), st.empty(), st.empty()
    speedup_slot = st.empty()
    st.caption("Streaming Workspace")
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    s_latency, s_ttft, s_tps, s_tpot = st.empty(), st.empty(), st.empty(), st.empty()
    spec_breakdown_slot = st.empty()
    st.caption("Streaming Workspace")
    spec_output_slot = st.empty()

# ==============================================================================
# PERSISTED RENDER LOGIC
# ==============================================================================

def display_persisted_metrics(base_res, spec_res, show_acceptance=False):
    if base_res:
        b_latency.metric("Total Request Time", f"{base_res['latency']:.2f}s")
        b_ttft.metric("Time to First Token (TTFT)", f"{base_res['ttft']:.3f}s")
        b_tps.metric("Output Tokens Per Second", f"{base_res['tokens_per_second']:.1f} tok/s")
        b_tpot.metric("Time Per Output Token (TPOT)", f"{base_res['tpot']:.1f} ms/tok")
        baseline_output_slot.markdown(base_res['text'])
    else:
        for slot in [b_latency, b_ttft, b_tps, b_tpot]: slot.metric("—", "—")

    if spec_res:
        s_latency.metric("Total Request Time", f"{spec_res['latency']:.2f}s")
        s_ttft.metric("Time to First Token (TTFT)", f"{spec_res['ttft']:.3f}s")
        s_tps.metric("Output Tokens Per Second", f"{spec_res['tokens_per_second']:.1f} tok/s")
        s_tpot.metric("Time Per Output Token (TPOT)", f"{spec_res['tpot']:.1f} ms/tok")
        spec_output_slot.markdown(spec_res['text'])
        if show_acceptance:
            spec_breakdown_slot.metric("Draft Acceptance Rate (This Run)", f"{spec_res['run_rate']:.1f}%")
        else:
            spec_breakdown_slot.empty()
    else:
        for slot in [s_latency, s_ttft, s_tps, s_tpot]: slot.metric("—", "—")
        spec_breakdown_slot.metric("Draft Acceptance Rate (This Run)", "—")

    if base_res and spec_res and spec_res['latency'] > 0:
        net_speedup = base_res['latency'] / spec_res['latency']
        speedup_slot.metric("Net Speedup Performance Delta", f"{net_speedup:.2f}x Faster")
    else:
        speedup_slot.metric("Net Speedup Performance Delta", "—")

# Render metrics based on frozen state tab configurations
if demo == "Demo 1: Workload Predictability":
    display_persisted_metrics(
        st.session_state.d1_state[active_key]["base_res"],
        st.session_state.d1_state[active_key]["spec_res"],
        show_acceptance=True
    )
elif demo == "Demo 2: Context Scaling":
    display_persisted_metrics(
        st.session_state.d2_state[context_tier]["base_res"],
        st.session_state.d2_state[context_tier]["spec_res"],
        show_acceptance=False # Irrelevant for the compute bounds lesson
    )

# ==============================================================================
# EXECUTION RACE THREAD LOOPER
# ==============================================================================

if execute_race:
    baseline_output_slot.info("Warming baseline runtime engine context...")
    spec_output_slot.info("Assembling speculative proposal matrix layers...")

    ctx = get_script_run_ctx()
    def run_with_runtime_context(client, p, max_tok, t, slot):
        add_script_run_ctx(threading.current_thread(), ctx)
        return stream_engine(client, p, max_tok, t, slot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_baseline = executor.submit(run_with_runtime_context, baseline_client, prompt, max_tokens, temperature, baseline_output_slot)
        future_spec = executor.submit(run_with_runtime_context, spec_client, prompt, max_tokens, temperature, spec_output_slot)
        
        while not (future_baseline.done() and future_spec.done()):
            time.sleep(0.1)

    base_res = future_baseline.result()
    spec_res = future_spec.result()

    # Save outputs to their designated state silos
    if demo == "Demo 1: Workload Predictability":
        st.session_state.d1_state[active_key]["base_res"] = base_res
        st.session_state.d1_state[active_key]["spec_res"] = spec_res
    elif demo == "Demo 2: Context Scaling":
        st.session_state.d2_state[context_tier]["base_res"] = base_res
        st.session_state.d2_state[context_tier]["spec_res"] = spec_res
        
    st.rerun()