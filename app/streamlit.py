import os
import time
import random
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

JSON_PROMPT = """Generate exactly 30 synthetic API request logs as a JSON array.
Each object must follow this schema exactly:
{
  "request_id": string,
  "endpoint": string,
  "method": "GET" | "POST" | "DELETE",
  "status_code": number,
  "latency_ms": number,
  "cached": boolean
}
Return only valid JSON with no markdown or commentary."""

CREATIVE_PROMPT = """Write a detailed fictional story of approximately 700–900 words about an engineer investigating a mysterious latency spike before a live production demo. Develop multiple plausible causes, include dialogue between engineers, and resolve the incident in an unexpected but technically plausible way."""

AUDIT_INSTRUCTION = """
[REGULATORY AUDIT MANDATE]
Analyze the audit records above.
Determine whether the CFO explicitly guaranteed investment yields.
Output exactly one concise sentence with your final compliance determination and cite the relevant timestamp.
"""

NEEDLE_RECORD = (
    "[TIMESTAMP 00:42:17] CFO OFFICE: During the quarterly risk review, the CFO stated "
    "that projected returns remain uncertain and explicitly declined to guarantee future yields. "
    "The committee recorded that all forward-looking return estimates remain subject to market, "
    "liquidity, currency, and counterparty risk."
)

DEPARTMENTS = ["Treasury Operations", "Internal Audit", "Capital Markets", "Risk Management", "Corporate Finance", "Liquidity Oversight", "Controller Office", "Investment Committee", "Regional Finance", "Compliance Review"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "United Kingdom", "Japan", "Southeast Asia"]
TOPICS = ["foreign exchange exposure", "regional liquidity reserves", "counterparty credit risk", "derivative positions", "short-term capital requirements", "settlement controls", "variable-rate instruments", "cash concentration balances", "market volatility exposure", "collateral requirements", "cross-border funding", "interest-rate sensitivity"]
ACTIONS = ["reviewed", "reconciled", "assessed", "validated", "stress-tested", "examined", "modeled", "monitored"]
OUTCOMES = [
    "No material exception was identified and the item remains subject to routine monitoring.",
    "The committee requested an updated sensitivity analysis for the next reporting cycle.",
    "Existing controls were retained pending the next scheduled review.",
    "The exposure remained within the approved internal risk threshold.",
    "A follow-up reconciliation was assigned to the regional finance team.",
    "The committee noted elevated uncertainty but made no change to the current control framework.",
]
TEMPLATES = [
    "[TIMESTAMP {ts}] {dept}: The team {action} {topic} associated with {region}. Reported exposure was approximately ${amount} million across {accounts} internal accounts. {outcome}",
    "[TIMESTAMP {ts}] {dept}: A scheduled review of {topic} in {region} was completed. Analysts compared current balances with prior-period assumptions and tested a {pct}% adverse scenario. {outcome}",
    "[TIMESTAMP {ts}] {dept}: Management {action} controls related to {topic} for operations in {region}. The working group evaluated {accounts} positions with an aggregate notional value near ${amount} million. {outcome}",
    "[TIMESTAMP {ts}] {dept}: The committee discussed {topic} affecting {region}. The analysis incorporated liquidity, settlement, and valuation assumptions under a {pct}% stress case. {outcome}",
]

def _audit_timestamp(i):
    total = (i * 47 + 83) % 86400
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

def generate_audit_record(i, rng):
    return rng.choice(TEMPLATES).format(
        ts=_audit_timestamp(i), dept=rng.choice(DEPARTMENTS), action=rng.choice(ACTIONS),
        topic=rng.choice(TOPICS), region=rng.choice(REGIONS), amount=rng.randint(4, 980),
        accounts=rng.randint(3, 240), pct=rng.randint(2, 35), outcome=rng.choice(OUTCOMES)
    )

def load_target_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    except Exception:
        return None

TOKENIZER = load_target_tokenizer()

def count_prompt_tokens(text):
    if TOKENIZER is not None:
        return len(TOKENIZER.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)

def build_audit_prompt(target_tokens, seed=42, evidence_fraction=0.70):
    rng = random.Random(seed + target_tokens)
    records = []
    i = 0
    reserve = count_prompt_tokens(AUDIT_INSTRUCTION) + count_prompt_tokens(NEEDLE_RECORD) + 100
    body_target = max(500, target_tokens - reserve)
    while count_prompt_tokens("\n".join(records)) < body_target:
        records.append(generate_audit_record(i, rng))
        i += 1
    insert_at = max(1, min(len(records) - 1, int(len(records) * evidence_fraction)))
    records.insert(insert_at, NEEDLE_RECORD)
    prompt = "\n".join(records) + AUDIT_INSTRUCTION
    return prompt, count_prompt_tokens(prompt), len(records), evidence_fraction

D2_TARGETS = {"8K Context": 8_000, "24K Context": 24_000, "32K Context": 32_000}
D2_PROMPTS = {}
D2_META = {}
for _label, _target in D2_TARGETS.items():
    _prompt, _estimated, _records, _position = build_audit_prompt(_target)
    D2_PROMPTS[_label] = _prompt
    D2_META[_label] = {"target_tokens": _target, "estimated_tokens": _estimated, "records": _records, "evidence_position": _position}

# ============================================================================== 
# SESSION STATE INITIALIZATION
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
    start_accepted = query_prometheus("sum(vllm:spec_decode_num_accepted_tokens_total)")
    start_drafted = query_prometheus("sum(vllm:spec_decode_num_draft_tokens_total)")

    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0
    prompt_token_count = None

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
                if hasattr(chunk.usage, "prompt_tokens"):
                    prompt_token_count = chunk.usage.prompt_tokens

        total_latency = time.perf_counter() - start
        generation_time = total_latency - (first_token_time or 0)
        if token_count == 0:
            token_count = len(text.split())

        tps = token_count / total_latency if total_latency > 0 else 0
        time.sleep(0.4)
        
        end_accepted = query_prometheus("sum(vllm:spec_decode_num_accepted_tokens_total)")
        end_drafted = query_prometheus("sum(vllm:spec_decode_num_draft_tokens_total)")
        run_accepted = end_accepted - start_accepted
        run_drafted = end_drafted - start_drafted
        run_rate = (run_accepted / run_drafted) * 100 if run_drafted > 0 else 0.0

        return {
            "latency": total_latency, "ttft": first_token_time, "tokens": token_count, 
            "prompt_tokens": prompt_token_count or count_prompt_tokens(prompt),
            "tokens_per_second": tps, "text": text, "run_rate": run_rate
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

if demo == "Demo 1: Workload Predictability":
    temperature = 0.0
    st.sidebar.caption("Temperature fixed at 0.0 for controlled comparison")
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
    temperature = 0.0  
    max_tokens = 128   
    context_tier = st.radio("Select Active Context Window Size", ["8K Context", "24K Context", "32K Context"], horizontal=True)
    prompt = D2_PROMPTS[context_tier]
    with st.expander("🔍 Inspect Synthetic Audit Context", expanded=True):
        meta = D2_META[context_tier]
        st.write(f"**Target context:** {meta['target_tokens']:,} tokens")
        st.write(f"**Construction token estimate:** {meta['estimated_tokens']:,} tokens")
        st.write(f"**Synthetic records:** {meta['records']:,}")
        st.write(f"**Controlled evidence position:** ~{meta['evidence_position']:.0%}")
        needle_idx = prompt.find(NEEDLE_RECORD)
        evidence_preview = prompt[max(0, needle_idx-250):needle_idx+len(NEEDLE_RECORD)+250] if needle_idx >= 0 else NEEDLE_RECORD
        st.text(prompt[:900] + "\n\n [... VARIED RECORDS OMITTED ...]\n\n" + evidence_preview + "\n\n [... VARIED RECORDS OMITTED ...]\n\n" + AUDIT_INSTRUCTION)

else:
    st.info("💡 **Demo 3 Instructions:** Transition to your live Grafana dashboard layout now. Fire up your external load generator to showcase concurrent scaling overhead.")
    st.stop()

st.divider()

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
    b_metric1, b_metric2, b_metric3, b_metric4, b_metric5 = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
    speedup_slot = st.empty()
    st.caption("Streaming Workspace")
    baseline_output_slot = st.empty()

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    s_metric1, s_metric2, s_metric3, s_metric4, s_metric5 = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
    spec_breakdown_slot = st.empty()
    st.caption("Streaming Workspace")
    spec_output_slot = st.empty()

# ==============================================================================
# CONDITIONAL METRIC RENDER LOGIC (TAILORED PER DEMO)
# ==============================================================================

def display_persisted_metrics(base_res, spec_res, current_demo):
    if current_demo == "Demo 1: Workload Predictability":
        # Demo 1: Acceptance Rate, Decode Throughput, E2E Latency (No TTFT, No TPOT)
        if base_res:
            b_metric1.metric("Total Request Time (E2E)", f"{base_res['latency']:.2f}s")
            b_metric2.metric("Decode Throughput", f"{base_res['tokens_per_second']:.1f} tok/s")
            b_metric3.empty()
            b_metric4.empty()
            b_metric5.empty()
            baseline_output_slot.markdown(base_res['text'])
        else:
            for slot in [b_metric1, b_metric2, b_metric3, b_metric4, b_metric5]: slot.empty()

        if spec_res:
            s_metric1.metric("Total Request Time (E2E)", f"{spec_res['latency']:.2f}s")
            s_metric2.metric("Decode Throughput", f"{spec_res['tokens_per_second']:.1f} tok/s")
            s_metric3.empty()
            s_metric4.empty()
            s_metric5.empty()
            spec_output_slot.markdown(spec_res['text'])
            spec_breakdown_slot.metric("Draft Acceptance Rate", f"{spec_res['run_rate']:.1f}%")
        else:
            for slot in [s_metric1, s_metric2, s_metric3, s_metric4, s_metric5]: slot.empty()
            spec_breakdown_slot.metric("Draft Acceptance Rate", "—")

    elif current_demo == "Demo 2: Context Scaling":
        # Demo 2: Actual Input Tokens, TTFT, Acceptance Rate, Decode Throughput, E2E Latency (No TPOT)
        if base_res:
            b_metric1.metric("Actual Input Tokens", f"{base_res['prompt_tokens']:,} tokens")
            b_metric2.metric("Time to First Token (TTFT)", f"{base_res['ttft']:.3f}s")
            b_metric3.metric("Total Request Time (E2E)", f"{base_res['latency']:.2f}s")
            b_metric4.metric("Decode Throughput", f"{base_res['tokens_per_second']:.1f} tok/s")
            b_metric5.empty()
            baseline_output_slot.markdown(base_res['text'])
        else:
            for slot in [b_metric1, b_metric2, b_metric3, b_metric4, b_metric5]: slot.empty()

        if spec_res:
            s_metric1.metric("Actual Input Tokens", f"{spec_res['prompt_tokens']:,} tokens")
            s_metric2.metric("Time to First Token (TTFT)", f"{spec_res['ttft']:.3f}s")
            s_metric3.metric("Total Request Time (E2E)", f"{spec_res['latency']:.2f}s")
            s_metric4.metric("Decode Throughput", f"{spec_res['tokens_per_second']:.1f} tok/s")
            s_metric5.empty()
            spec_output_slot.markdown(spec_res['text'])
            spec_breakdown_slot.metric("Draft Acceptance Rate", f"{spec_res['run_rate']:.1f}%")
        else:
            for slot in [s_metric1, s_metric2, s_metric3, s_metric4, s_metric5]: slot.empty()
            spec_breakdown_slot.metric("Draft Acceptance Rate", "—")

    # Net Speedup evaluation box
    if base_res and spec_res and spec_res['latency'] > 0:
        net_speedup = base_res['latency'] / spec_res['latency']
        speedup_slot.metric("End-to-End Speculative Speedup", f"{net_speedup:.2f}x")
        if net_speedup > 1.05:
            st.success(f"Speculative decoding helped: {net_speedup:.2f}× lower end-to-end request time.")
        elif net_speedup < 0.95:
            st.warning(f"Speculative decoding hurt performance: {net_speedup:.2f}× baseline/spec ratio.")
        else:
            st.info(f"Speculative decoding was roughly neutral: {net_speedup:.2f}×.")
    else:
        speedup_slot.metric("End-to-End Speculative Speedup", "—")

# Render metrics based on active session tier
if demo == "Demo 1: Workload Predictability":
    display_persisted_metrics(
        st.session_state.d1_state[active_key]["base_res"],
        st.session_state.d1_state[active_key]["spec_res"],
        demo
    )
elif demo == "Demo 2: Context Scaling":
    display_persisted_metrics(
        st.session_state.d2_state[context_tier]["base_res"],
        st.session_state.d2_state[context_tier]["spec_res"],
        demo
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

    if demo == "Demo 1: Workload Predictability":
        st.session_state.d1_state[active_key]["base_res"] = base_res
        st.session_state.d1_state[active_key]["spec_res"] = spec_res
    elif demo == "Demo 2: Context Scaling":
        st.session_state.d2_state[context_tier]["base_res"] = base_res
        st.session_state.d2_state[context_tier]["spec_res"] = spec_res
    st.rerun()