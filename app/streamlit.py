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

# --- REAL ENTERPRISE COMPLIANCE PROMPT LAYER ---
COMPLIANCE_TEXT_BLOCK = """
[TIMESTAMP 00:14:22] AUDIT RECORD EXCERPT: The entity's primary fiscal steering committee reviewed unhedged international volatile asset surfaces. Operational capital preserves were systematically re-allocated across regional holdings to buffer liquidity degradation vectors. The CFO affirmed during the live presentation that while foreign exchange metrics show persistent downward pressure, our current exposure matrix maintains a highly predictable, risk-adjusted profile without relying on artificial valuation multipliers. Moving on to secondary market derivatives...
"""

INSTRUCTION_BLOCK = """
\n\n[REGULATORY AUDIT MANDATE]
Analyze the entire transaction audit logging history above. 
Verify if the financial executor explicitly guaranteed yields or used the phrase 'guaranteed yield' regarding volatile asset classes.
Output exactly one concise sentence declaring your final compliance determination and cite the section timestamp.
"""

# Build real scaling prompts using exact text repetitions so the audience sees the scale
PROMPTS = {
    "8K Context": (COMPLIANCE_TEXT_BLOCK * 70) + INSTRUCTION_BLOCK,
    "24K Context": (COMPLIANCE_TEXT_BLOCK * 210) + INSTRUCTION_BLOCK,
    "32K Context": (COMPLIANCE_TEXT_BLOCK * 280) + INSTRUCTION_BLOCK
}

# Initialize multi-tab persistence states
if "d2_state" not in st.session_state:
    st.session_state.d2_state = {
        "8K Context": {"base_res": None, "spec_res": None},
        "24K Context": {"base_res": None, "spec_res": None},
        "32K Context": {"base_res": None, "spec_res": None}
    }

def query_prometheus(query: str):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0

def stream_engine(client, prompt: str, max_tokens: int, output_slot):
    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0
    try:
        stream = client.chat.completions.create(
            model=TARGET_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, stream=True,
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

        return {
            "latency": total_latency, "ttft": first_token_time, "tokens": token_count,
            "tokens_per_second": token_count / total_latency if total_latency > 0 else 0,
            "tpot": (generation_time / token_count) * 1000 if token_count > 0 else 0, "text": text
        }
    except Exception as e:
        output_slot.error(f"Inference Fault: {e}")
        return None

st.set_page_config(page_title="vLLM Context Arena", layout="wide")
st.title("⚡ Demo 2: Context Window Scaling Bounds")

# Create selection for the 3 distinct context tiers
context_tier = st.radio("Select Active Context Window Size", ["8K Context", "24K Context", "32K Context"], horizontal=True)

with st.expander("🔍 View Active Compliance Prompt Footprint Running on GPU", expanded=False):
    st.write(f"**Total Document Payload Footprint:** ~{context_tier}")
    st.text(PROMPTS[context_tier][:1200] + "\n\n [... MIDDLE OMITTED FOR VIEWPORT ...]\n\n" + INSTRUCTION_BLOCK)

st.divider()

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    execute_race = st.button("🚀 Run Scaled Context Race", type="primary")
with col_btn2:
    if st.button("Clear Context Matrices"):
        st.session_state.d2_state = {k: {"base_res": None, "spec_res": None} for k in PROMPTS.keys()}
        st.rerun()

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.subheader("🤖 Traditional Baseline Engine")
    b_latency, b_ttft, b_tps, b_tpot = st.empty(), st.empty(), st.empty(), st.empty()
    speedup_slot = st.empty()
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    s_latency, s_ttft, s_tps, s_tpot = st.empty(), st.empty(), st.empty(), st.empty()
    spec_output_slot = st.empty()

def render_metrics(base, spec):
    if base:
        b_latency.metric("Total Request Time", f"{base['latency']:.2f}s")
        b_ttft.metric("Time to First Token (TTFT - Prefill)", f"{base['ttft']:.3f}s")
        b_tps.metric("Output Tokens Per Second", f"{base['tokens_per_second']:.1f} tok/s")
        b_tpot.metric("Time Per Output Token (TPOT - Decode)", f"{base['tpot']:.1f} ms/tok")
        baseline_output_slot.markdown(base['text'])
    else:
        for slot in [b_latency, b_ttft, b_tps, b_tpot]: slot.metric("—", "—")
        
    if spec:
        s_latency.metric("Total Request Time", f"{spec['latency']:.2f}s")
        s_ttft.metric("Time to First Token (TTFT - Prefill)", f"{spec['ttft']:.3f}s")
        s_tps.metric("Output Tokens Per Second", f"{spec['tokens_per_second']:.1f} tok/s")
        s_tpot.metric("Time Per Output Token (TPOT - Decode)", f"{spec['tpot']:.1f} ms/tok")
        spec_output_slot.markdown(spec['text'])
    else:
        for slot in [s_latency, s_ttft, s_tps, s_tpot]: slot.metric("—", "—")

    if base and spec:
        net_speedup = base['latency'] / spec['latency']
        speedup_slot.metric("Net Speedup Performance Delta", f"{net_speedup:.2f}x Faster")
    else:
        speedup_slot.metric("Net Speedup Performance Delta", "—")

# Display frozen metrics if they exist for this specific context tab
render_metrics(st.session_state.d2_state[context_tier]["base_res"], st.session_state.d2_state[context_tier]["spec_res"])

if execute_race:
    baseline_output_slot.info("Executing standard baseline model prefill phase...")
    spec_output_slot.info("Executing speculative acceleration model prefill phase...")
    
    ctx = get_script_run_ctx()
    def run_with_ctx(client, p, max_tok, slot):
        add_script_run_ctx(threading.current_thread(), ctx)
        return stream_engine(client, p, max_tok, slot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_base = executor.submit(run_with_ctx, baseline_client, PROMPTS[context_tier], 128, baseline_output_slot)
        f_spec = executor.submit(run_with_ctx, spec_client, PROMPTS[context_tier], 128, spec_output_slot)
        while not (f_base.done() and f_spec.done()): time.sleep(0.1)

    st.session_state.d2_state[context_tier]["base_res"] = f_base.result()
    st.session_state.d2_state[context_tier]["spec_res"] = f_spec.result()
    st.rerun()