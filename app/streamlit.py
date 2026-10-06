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

EXTRACTION_PROMPT = """Extract the incident information from the text below and return it as valid JSON.
Use exactly this schema:
{
  "service": string,
  "region": string,
  "incident_type": string,
  "severity": "low" | "medium" | "high" | "critical",
  "error_code": string,
  "duration_minutes": number,
  "resolved": boolean
}
Incident Report:
At 14:32 UTC, the payment-api service in the us-west-2 region began experiencing elevated request failures caused by database connection exhaustion. The incident was classified as high severity and produced error code DB_CONN_503. Engineers restored normal service after 27 minutes by increasing the connection pool capacity. The incident is now fully resolved.
Return only the JSON object. Do not include markdown, explanations, or additional fields."""

CREATIVE_PROMPT = """Write a detailed fictional story of approximately 700–900 words about an engineer investigating a mysterious latency spike before a live production demo. Develop multiple plausible causes, include dialogue between engineers, and resolve the incident in an unexpected but technically plausible way."""

BRAINSTORM_PROMPT = """A software company wants to redesign how engineers respond to unexpected production incidents.
Brainstorm 15 distinct ideas that could make incident response faster, less stressful, and more effective.
Explore a wide range of possibilities, including improvements to developer tools, monitoring, team communication, automation, documentation, and organizational processes. Avoid giving minor variations of the same idea.
For each idea, briefly explain how it would work and why it could improve incident response."""

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

# ==============================================================================
# CACHED TOKENIZER & LAZY PROMPT GENERATION
# ==============================================================================

@st.cache_resource
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

@st.cache_data
def get_d2_prompt(context_label):
    D2_TARGETS = {"8K": 8_000, "24K": 24_000, "32K": 32_000}
    target = D2_TARGETS[context_label]
    prompt, estimated, records, position = build_audit_prompt(target)
    meta = {
        "target_tokens": target, 
        "estimated_tokens": estimated, 
        "records": records, 
        "evidence_position": position
    }
    return prompt, meta

# ============================================================================== 
# SESSION STATE INITIALIZATION
# ============================================================================== 

if "d1_state" not in st.session_state:
    st.session_state.d1_state = {
        "JSON Generation": {"base_res": None, "spec_res": None},
        "Structured Extraction": {"base_res": None, "spec_res": None},
        "Creative Story": {"base_res": None, "spec_res": None},
        "Brainstorming Incident Response": {"base_res": None, "spec_res": None}
    }

if "d2_state" not in st.session_state:
    st.session_state.d2_state = {
        "8K": {"base_res": None, "spec_res": None},
        "24K": {"base_res": None, "spec_res": None},
        "32K": {"base_res": None, "spec_res": None}
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
            "tokens_per_second": tps, "text": text, "run_rate": run_rate,
            "run_accepted": int(run_accepted), "run_drafted": int(run_drafted)
        }
    except Exception as e:
        output_slot.error(f"Inference Fault: {e}")
        return None

# ==============================================================================
# UI SETUP & ROUTING
# ==============================================================================

st.set_page_config(page_title="vLLM Inference Optimization Arena", layout="wide")
st.title("vLLM Live Inference Optimization Arena")

st.sidebar.header("Navigation")
demo = st.sidebar.radio("Select Demo Scenario", ["Demo 1: Workload Predictability", "Demo 2: Context Scaling", "Demo 3: Production Stress Note"])

if demo == "Demo 1: Workload Predictability":
    temperature = 0.0
    st.sidebar.caption("Temperature fixed at 0.0 for controlled comparison")
    max_tokens = st.sidebar.slider("Max Output Tokens", 128, 1024, 384, step=64)
    
    st.markdown("### DEMO 1 — WORKLOAD PREDICTABILITY")
    workload_type = st.radio("Workload Type:", ["📋 Predictable", "🎨 Open-Ended"], horizontal=True)
    
    if workload_type == "📋 Predictable":
        active_key = st.selectbox("Example:", ["JSON Generation", "Structured Extraction"])
        prompt = JSON_PROMPT if active_key == "JSON Generation" else EXTRACTION_PROMPT
    else:
        active_key = st.selectbox("Example:", ["Creative Story", "Brainstorming Incident Response"])
        prompt = CREATIVE_PROMPT if active_key == "Creative Story" else BRAINSTORM_PROMPT
        
    with st.expander("🔍 View Prompt Running on Stage", expanded=True):
        st.code(prompt, language="text")

elif demo == "Demo 2: Context Scaling":
    temperature = 0.0  
    max_tokens = 128   
    
    st.markdown("### DEMO 2 — CONTEXT SCALING")
    context_tier = st.radio("Context:", ["8K", "24K", "32K"], horizontal=True)
    
    prompt, meta = get_d2_prompt(context_tier)
    st.markdown(f"**Actual Input Tokens:** `{meta['estimated_tokens']:,}`")
    
    with st.expander("🔍 Inspect Synthetic Audit Context", expanded=True):
        st.write(f"**Target context:** {meta['target_tokens']:,} tokens")
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
    execute_race = st.button("Run Experiment", type="primary")
with col_btn2:
    if st.button("Clear Playback Matrices"):
        if demo == "Demo 1: Workload Predictability":
            st.session_state.d1_state = {k: {"base_res": None, "spec_res": None} for k in st.session_state.d1_state}
        else:
            st.session_state.d2_state = {k: {"base_res": None, "spec_res": None} for k in st.session_state.d2_state}
        st.rerun()

# ==============================================================================
# RENDER LOGIC (SHARED ENGINE DISPLAY FORMAT)
# ==============================================================================

if demo == "Demo 1: Workload Predictability":
    base_res = st.session_state.d1_state[active_key]["base_res"]
    spec_res = st.session_state.d1_state[active_key]["spec_res"]
elif demo == "Demo 2: Context Scaling":
    base_res = st.session_state.d2_state[context_tier]["base_res"]
    spec_res = st.session_state.d2_state[context_tier]["spec_res"]

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.subheader("🤖 Traditional Baseline Engine")
    st.markdown("---")
    
    st.markdown("**TTFT**")
    st.markdown(f"### `{base_res['ttft']:.3f}s`" if base_res and base_res.get('ttft') is not None else "`—`")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Decode Throughput**")
    st.markdown(f"### `{base_res['tokens_per_second']:.1f} tok/s`" if base_res else "`—`")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**End-to-End Latency**")
    st.markdown(f"### `{base_res['latency']:.2f}s`" if base_res else "`—`")

with metric_col2:
    st.subheader("🚀 Speculative Accelerated Engine")
    st.markdown("---")
    
    st.markdown("**TTFT**")
    if base_res and spec_res and base_res.get('ttft') and spec_res.get('ttft'):
        ttft_diff = ((spec_res['ttft'] - base_res['ttft']) / base_res['ttft']) * 100
        st.markdown(f"### `{spec_res['ttft']:.3f}s`")
        if ttft_diff > 0:
            st.warning(f"↑ {ttft_diff:.0f}% slower")
        else:
            st.success(f"↓ {abs(ttft_diff):.0f}% faster")
    elif spec_res and spec_res.get('ttft'):
        st.markdown(f"### `{spec_res['ttft']:.3f}s`")
    else:
        st.markdown("`—`")

    st.markdown("**Decode Throughput**")
    if base_res and spec_res:
        speedup_val = spec_res['tokens_per_second'] / max(0.1, base_res['tokens_per_second'])
        pct_increase = ((spec_res['tokens_per_second'] - base_res['tokens_per_second']) / max(0.1, base_res['tokens_per_second'])) * 100
        st.markdown(f"### `{spec_res['tokens_per_second']:.1f} tok/s`")
        st.success(f"↑ {speedup_val:.2f}× (+{pct_increase:.0f}%)")
    elif spec_res:
        st.markdown(f"### `{spec_res['tokens_per_second']:.1f} tok/s`")
    else:
        st.markdown("`—`")

    st.markdown("**End-to-End Latency**")
    if base_res and spec_res:
        lat_reduction = ((base_res['latency'] - spec_res['latency']) / max(0.001, base_res['latency'])) * 100
        st.markdown(f"### `{spec_res['latency']:.2f}s`")
        if lat_reduction > 0:
            st.info(f"↓ {lat_reduction:.0f}%")
        else:
            st.warning(f"↑ {abs(lat_reduction):.0f}% slower")
    elif spec_res:
        st.markdown(f"### `{spec_res['latency']:.2f}s`")
    else:
        st.markdown("`—`")

    st.markdown("**Draft Acceptance Rate**")
    st.markdown(f"### `{spec_res['run_rate']:.1f}%`" if spec_res else "`—`")

st.divider()

# Advanced Speculative Metrics Expander
with st.expander("▼ Advanced Speculative Metrics"):
    if spec_res:
        acc_tokens = spec_res.get('run_accepted', 0)
        draft_tokens = spec_res.get('run_drafted', 0)
        steps = max(1, draft_tokens // 4)
        mean_len = (1 + (acc_tokens / steps)) if steps > 0 else 0.0
        
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        col_adv1.metric("Mean Accepted Tokens / Step", f"{mean_len:.2f}")
        col_adv2.metric("Accepted Draft Tokens", f"{acc_tokens:,}")
        col_adv3.metric("Proposed Draft Tokens", f"{draft_tokens:,}")
    else:
        st.info("_Run an experiment to populate speculative telemetry metrics._")

st.divider()

# Response Output Workspaces (Rendered AFTER all metrics & expanders)
st.markdown("### MODEL OUTPUTS")
out_col1, out_col2 = st.columns(2)
with out_col1:
    st.caption("🤖 Baseline Output")
    baseline_output_slot = st.empty()
    if base_res:
        baseline_output_slot.markdown(base_res['text'])
    else:
        baseline_output_slot.markdown("_Awaiting run..._")

with out_col2:
    st.caption("🚀 Speculative Output")
    spec_output_slot = st.empty()
    if spec_res:
        spec_output_slot.markdown(spec_res['text'])
    else:
        spec_output_slot.markdown("_Awaiting run..._")

# ==============================================================================
# EXECUTION RACE THREAD LOOPER
# ==============================================================================

if execute_race:
    temp_slot_b = st.empty()
    temp_slot_s = st.empty()
    temp_slot_b.info("Warming baseline runtime engine context...")
    temp_slot_s.info("Assembling speculative proposal matrix layers...")

    ctx = get_script_run_ctx()
    def run_with_runtime_context(client, p, max_tok, t, slot):
        add_script_run_ctx(threading.current_thread(), ctx)
        return stream_engine(client, p, max_tok, t, slot)

    dummy_slot_b = st.empty()
    dummy_slot_s = st.empty()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_baseline = executor.submit(run_with_runtime_context, baseline_client, prompt, max_tokens, temperature, dummy_slot_b)
        future_spec = executor.submit(run_with_runtime_context, spec_client, prompt, max_tokens, temperature, dummy_slot_s)
        while not (future_baseline.done() and future_spec.done()):
            time.sleep(0.1)

    temp_slot_b.empty()
    temp_slot_s.empty()

    base_res = future_baseline.result()
    spec_res = future_spec.result()

    if demo == "Demo 1: Workload Predictability":
        st.session_state.d1_state[active_key]["base_res"] = base_res
        st.session_state.d1_state[active_key]["spec_res"] = spec_res
    elif demo == "Demo 2: Context Scaling":
        st.session_state.d2_state[context_tier]["base_res"] = base_res
        st.session_state.d2_state[context_tier]["spec_res"] = spec_res
    st.rerun()