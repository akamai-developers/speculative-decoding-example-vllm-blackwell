import os
import time
import random
import statistics
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

BASELINE_URL = "http://127.0.0.1:8000/v1"
SPEC_URL = "http://127.0.0.1:8001/v1"
TARGET_MODEL = os.getenv("TARGET_MODEL")
NUM_SPEC_TOKENS = int(os.getenv("NUM_SPEC_TOKENS", "5"))

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
    # Grow in batches and count occasionally; re-tokenizing every step is slow at 32K.
    while True:
        for _ in range(20):
            records.append(generate_audit_record(i, rng))
            i += 1
        if count_prompt_tokens("\n".join(records)) >= body_target:
            break
    while len(records) > 1 and count_prompt_tokens("\n".join(records[:-1])) >= body_target:
        records.pop()
    insert_at = max(1, min(len(records) - 1, int(len(records) * evidence_fraction)))
    records.insert(insert_at, NEEDLE_RECORD)
    prompt = "\n".join(records) + AUDIT_INSTRUCTION
    return prompt, count_prompt_tokens(prompt), len(records), evidence_fraction

D2_TARGETS = {"1K": 1_000, "8K": 8_000, "16K": 16_000, "24K": 24_000, "32K": 32_000}

@st.cache_data
def get_d2_prompt(context_label):
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

# Results are keyed by experiment (e.g. "JSON Generation @ T=0.0"), so changing a
# setting never overwrites an earlier run.
if "results" not in st.session_state:
    st.session_state.results = {}

if "d3_results" not in st.session_state:
    st.session_state.d3_results = None

# ==============================================================================
# vLLM METRICS
# ==============================================================================

ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED_METRIC = "vllm:spec_decode_num_draft_tokens_total"
DRAFTS_METRIC = "vllm:spec_decode_num_drafts_total"   # number of draft (verification) steps

def get_vllm_counters(names) -> dict:
    """Sum each counter across label sets from the speculative server's /metrics."""
    totals = {n: 0.0 for n in names}
    try:
        metrics_url = f"{SPEC_URL.replace('/v1', '')}/metrics"
        response = requests.get(metrics_url, timeout=2)
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("#"):
                continue
            for n in names:
                # Exact metric name, followed by labels or a space (not a longer name).
                if line.startswith(n + "{") or line.startswith(n + " "):
                    try:
                        totals[n] += float(line.rsplit(" ", 1)[-1])
                    except ValueError:
                        pass
    except Exception:
        pass
    return totals

SPEC_COUNTERS = [ACCEPTED_METRIC, DRAFTED_METRIC, DRAFTS_METRIC]

def spec_stats(before: dict, after: dict) -> dict:
    accepted = after[ACCEPTED_METRIC] - before[ACCEPTED_METRIC]
    drafted = after[DRAFTED_METRIC] - before[DRAFTED_METRIC]
    steps = after[DRAFTS_METRIC] - before[DRAFTS_METRIC]
    if steps <= 0 and drafted > 0:
        steps = drafted / NUM_SPEC_TOKENS   # fallback if this vLLM version lacks num_drafts
    return {
        "run_accepted": int(accepted),
        "run_drafted": int(drafted),
        "run_steps": int(round(steps)),
        "run_rate": (accepted / drafted) * 100 if drafted > 0 else 0.0,
        # Mean acceptance length: tokens gained per verification step = 1 + accepted/steps.
        "mean_accept_len": (1 + accepted / steps) if steps > 0 else 0.0,
    }

# ==============================================================================
# SINGLE REQUEST (streaming, so we can measure TTFT and TPOT)
# ==============================================================================

def stream_engine(client, prompt, max_tokens, temp, output_slot=None, fixed_length=False, is_spec=False):
    before = get_vllm_counters(SPEC_COUNTERS) if is_spec else None

    start = time.perf_counter()
    first_token_time = None
    text = ""
    token_count = 0
    prompt_token_count = None
    extra = {"ignore_eos": True} if fixed_length else {}

    stream = client.chat.completions.create(
        model=TARGET_MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temp, stream=True,
        stream_options={"include_usage": True}, extra_body=extra or None,
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                text += delta
                if output_slot is not None:
                    output_slot.markdown(text)
        if getattr(chunk, "usage", None) is not None:
            token_count = chunk.usage.completion_tokens
            prompt_token_count = chunk.usage.prompt_tokens

    total_latency = time.perf_counter() - start
    if token_count == 0:
        token_count = len(text.split())
    if first_token_time is None:
        first_token_time = total_latency

    # Decode-phase metrics: everything after the first token.
    decode_time = max(0.0, total_latency - first_token_time)
    tpot = decode_time / (token_count - 1) if token_count > 1 else 0.0

    result = {
        "latency": total_latency,
        "ttft": first_token_time,
        "tpot_ms": tpot * 1000,
        "decode_tps": (1 / tpot) if tpot > 0 else 0.0,
        "e2e_tps": token_count / total_latency if total_latency > 0 else 0.0,
        "tokens": token_count,
        "prompt_tokens": prompt_token_count or count_prompt_tokens(prompt),
        "text": text,
    }
    if is_spec:
        time.sleep(0.3)  # let the counters settle
        result.update(spec_stats(before, get_vllm_counters(SPEC_COUNTERS)))
    return result

def warm_up(client):
    """One short request so the first measured run doesn't include start-up costs."""
    try:
        client.chat.completions.create(
            model=TARGET_MODEL, messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=8, temperature=0.0,
        )
    except Exception:
        pass

NUMERIC_KEYS = ["latency", "ttft", "tpot_ms", "decode_tps", "e2e_tps", "tokens", "prompt_tokens",
                "run_accepted", "run_drafted", "run_steps", "run_rate", "mean_accept_len"]

def median_result(runs):
    if not runs:
        return None
    out = {"text": runs[-1]["text"], "n_runs": len(runs)}
    for k in NUMERIC_KEYS:
        vals = [r[k] for r in runs if k in r and r[k] is not None]
        if vals:
            out[k] = statistics.median(vals)
    return out

# ==============================================================================
# UI SETUP & ROUTING
# ==============================================================================

st.set_page_config(page_title="vLLM Speculative Decoding Benchmark", layout="wide")
st.title("vLLM Speculative Decoding Benchmark")

st.sidebar.header("Navigation")
demo = st.sidebar.radio("Select Demo Scenario", [
    "Demo 1: Workload Predictability",
    "Demo 2: Context Scaling",
    "Demo 3: Concurrency Scaling"
])

st.sidebar.divider()
st.sidebar.header("Run settings")
repeats = st.sidebar.slider("Repeats per server (median reported)", 1, 5, 3)
do_warmup = st.sidebar.checkbox("Warm up each server before measuring", value=True)
st.sidebar.caption(
    f"Servers are measured **one at a time** (baseline, then speculative), so they never "
    f"compete for the GPU. Draft tokens per step (k): **{NUM_SPEC_TOKENS}**."
)

if demo == "Demo 1: Workload Predictability":
    st.markdown("### DEMO 1 — WORKLOAD PREDICTABILITY")
    temperature = st.sidebar.select_slider("Temperature", options=[0.0, 0.3, 0.5, 0.7, 1.0], value=0.0)
    max_tokens = st.sidebar.slider("Max Output Tokens", 128, 1024, 384, step=64)
    fixed_length = st.sidebar.checkbox(
        "Fixed output length (ignore EOS)", value=False,
        help="Forces every response to exactly Max Output Tokens, so output length is identical across tasks. "
             "TPOT is already length-independent, so this mainly matters for end-to-end latency."
    )

    workload_type = st.radio("Workload Type:", ["Predictable", "Open-Ended"], horizontal=True)
    if workload_type == "Predictable":
        task = st.selectbox("Example:", ["JSON Generation", "Structured Extraction"])
        prompt = JSON_PROMPT if task == "JSON Generation" else EXTRACTION_PROMPT
    else:
        task = st.selectbox("Example:", ["Creative Story", "Brainstorming Incident Response"])
        prompt = CREATIVE_PROMPT if task == "Creative Story" else BRAINSTORM_PROMPT
    active_key = f"D1 · {task} · T={temperature} · max={max_tokens}{' · fixed' if fixed_length else ''}"

    with st.expander("View Prompt Running on Stage", expanded=True):
        st.code(prompt, language="text")

elif demo == "Demo 2: Context Scaling":
    st.markdown("### DEMO 2 — CONTEXT SCALING")
    temperature = 0.0
    max_tokens = st.sidebar.slider("Output Tokens", 64, 512, 128, step=64)
    fixed_length = st.sidebar.checkbox(
        "Fixed output length (ignore EOS)", value=True,
        help="Keeps the output identical across context sizes, so input length is the only variable."
    )
    context_tier = st.radio("Context:", list(D2_TARGETS.keys()), horizontal=True)

    prompt, meta = get_d2_prompt(context_tier)
    active_key = f"D2 · {context_tier} · out={max_tokens}{' · fixed' if fixed_length else ''}"
    st.markdown(f"**Actual Input Tokens:** `{meta['estimated_tokens']:,}`")

    with st.expander("Inspect Synthetic Audit Context", expanded=True):
        st.write(f"**Target context:** {meta['target_tokens']:,} tokens")
        st.write(f"**Synthetic records:** {meta['records']:,}")
        st.write(f"**Controlled evidence position:** ~{meta['evidence_position']:.0%}")
        needle_idx = prompt.find(NEEDLE_RECORD)
        evidence_preview = prompt[max(0, needle_idx-250):needle_idx+len(NEEDLE_RECORD)+250] if needle_idx >= 0 else NEEDLE_RECORD
        st.text(prompt[:900] + "\n\n [... VARIED RECORDS OMITTED ...]\n\n" + evidence_preview + "\n\n [... VARIED RECORDS OMITTED ...]\n\n" + AUDIT_INSTRUCTION)

    # Summary across all context tiers run so far
    d2_rows = []
    for tier in D2_TARGETS:
        key = f"D2 · {tier} · out={max_tokens}{' · fixed' if fixed_length else ''}"
        r = st.session_state.results.get(key)
        if r and r.get("base") and r.get("spec"):
            b, s = r["base"], r["spec"]
            d2_rows.append({
                "Context": tier,
                "Input tokens": int(b["prompt_tokens"]),
                "Base TTFT (s)": round(b["ttft"], 3), "Spec TTFT (s)": round(s["ttft"], 3),
                "Base TPOT (ms)": round(b["tpot_ms"], 2), "Spec TPOT (ms)": round(s["tpot_ms"], 2),
                "Decode speedup": f"{b['tpot_ms'] / max(1e-6, s['tpot_ms']):.2f}x",
                "E2E speedup": f"{b['latency'] / max(1e-6, s['latency']):.2f}x",
                "Acceptance": f"{s.get('run_rate', 0):.1f}%",
            })
    if d2_rows:
        with st.expander("All context tiers so far", expanded=False):
            st.table(pd.DataFrame(d2_rows))

else:
    # ==========================================================================
    # DEMO 3 — CONCURRENCY
    # ==========================================================================
    st.markdown("### DEMO 3 — CONCURRENCY SCALING")
    st.markdown("**How does increasing server load affect the benefit of speculative decoding?**")
    st.caption("Start the servers with `./scripts/start_vllm.sh demo3` for this demo.")

    levels = [1, 2, 4, 8, 16, 32]
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f"**Concurrency levels:** {', '.join(map(str, levels))}")
    with col_c2:
        requests_per_test = st.slider("Requests per level:", 16, 128, 64, step=16)
    with col_c3:
        d3_max_tokens = st.slider("Output tokens per request:", 64, 512, 256, step=64)
    d3_fixed = st.checkbox("Fixed output length (ignore EOS)", value=True)

    run_concurrency_btn = st.button("RUN CONCURRENCY TEST", type="primary")

    def fire_request(client, p):
        """Streaming request so we get per-request TTFT and TPOT."""
        try:
            return stream_engine(client, p, d3_max_tokens, 0.0, None, fixed_length=d3_fixed)
        except Exception:
            return None

    def run_load(client, c, is_spec):
        before = get_vllm_counters(SPEC_COUNTERS) if is_spec else None
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as executor:
            results = list(executor.map(lambda _: fire_request(client, JSON_PROMPT), range(requests_per_test)))
        duration = time.perf_counter() - start
        ok = [r for r in results if r]
        out = {
            "duration": duration,
            "ok": len(ok),
            "out_tps": sum(r["tokens"] for r in ok) / duration if duration > 0 else 0.0,
            "req_s": len(ok) / duration if duration > 0 else 0.0,
            "tpot_ms": statistics.median([r["tpot_ms"] for r in ok]) if ok else 0.0,
            "ttft": statistics.median([r["ttft"] for r in ok]) if ok else 0.0,
        }
        if is_spec:
            time.sleep(0.3)
            out.update(spec_stats(before, get_vllm_counters(SPEC_COUNTERS)))
        return out

    if run_concurrency_btn:
        results_data = []
        status_box = st.status("Running concurrency test...", expanded=True)
        if do_warmup:
            status_box.write("Warming up both servers...")
            warm_up(baseline_client)
            warm_up(spec_client)

        for c in levels:
            status_box.update(label=f"Concurrency {c}: baseline...")
            b = run_load(baseline_client, c, is_spec=False)
            status_box.update(label=f"Concurrency {c}: speculative...")
            s = run_load(spec_client, c, is_spec=True)
            status_box.write(f"Concurrency {c} done.")

            results_data.append({
                "Concurrency": c,
                "Baseline out tok/s": round(b["out_tps"], 1),
                "Spec out tok/s": round(s["out_tps"], 1),
                "Throughput gain": f"{s['out_tps'] / max(1e-6, b['out_tps']):.2f}x",
                "Baseline TPOT (ms)": round(b["tpot_ms"], 2),
                "Spec TPOT (ms)": round(s["tpot_ms"], 2),
                "Spec acceptance": f"{s.get('run_rate', 0):.1f}%",
                "Failed requests (b/s)": f"{requests_per_test - b['ok']}/{requests_per_test - s['ok']}",
            })

        status_box.update(label="Concurrency test complete.", state="complete", expanded=False)
        st.session_state.d3_results = results_data

    st.markdown("### Concurrency Scaling Results")
    if st.session_state.d3_results:
        df_res = pd.DataFrame(st.session_state.d3_results)
        st.table(df_res)
        st.line_chart(
            df_res.set_index("Concurrency")[["Baseline out tok/s", "Spec out tok/s"]],
            x_label="Concurrent requests", y_label="Output tokens / s",
        )
        st.download_button("Download results (CSV)", df_res.to_csv(index=False), "demo3_results.csv")
    else:
        st.info("Click **RUN CONCURRENCY TEST** to measure both servers. Baseline and speculative "
                "are tested one after the other at each concurrency level.")

    st.stop()

# ==============================================================================
# DEMOS 1–2: RUN + DISPLAY
# ==============================================================================

st.divider()

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    execute_run = st.button("Run Experiment", type="primary")
with col_btn2:
    if st.button("Clear Results"):
        st.session_state.results = {}
        st.rerun()

st.markdown("### MODEL OUTPUTS")
out_col1, out_col2 = st.columns(2)
with out_col1:
    st.caption("Baseline Output")
    baseline_output_slot = st.empty()
with out_col2:
    st.caption("Speculative Output")
    spec_output_slot = st.empty()

if execute_run:
    status = st.status("Running...", expanded=True)
    try:
        if do_warmup:
            status.write("Warming up both servers...")
            warm_up(baseline_client)
            warm_up(spec_client)

        base_runs, spec_runs = [], []
        for i in range(repeats):
            # Strictly sequential: the speculative request starts only after the baseline finishes.
            status.update(label=f"Run {i+1}/{repeats}: baseline...")
            base_runs.append(stream_engine(baseline_client, prompt, max_tokens, temperature,
                                           baseline_output_slot, fixed_length=fixed_length))
            status.update(label=f"Run {i+1}/{repeats}: speculative...")
            spec_runs.append(stream_engine(spec_client, prompt, max_tokens, temperature,
                                           spec_output_slot, fixed_length=fixed_length, is_spec=True))

        st.session_state.results[active_key] = {
            "base": median_result(base_runs),
            "spec": median_result(spec_runs),
        }
        status.update(label="Done.", state="complete", expanded=False)
    except Exception as e:
        status.update(label=f"Inference error: {e}", state="error")

res = st.session_state.results.get(active_key, {})
base_res, spec_res = res.get("base"), res.get("spec")

if base_res and not execute_run:
    baseline_output_slot.markdown(base_res["text"])
elif not base_res:
    baseline_output_slot.markdown("_Awaiting run..._")
if spec_res and not execute_run:
    spec_output_slot.markdown(spec_res["text"])
elif not spec_res:
    spec_output_slot.markdown("_Awaiting run..._")

st.divider()
if base_res:
    st.caption(f"Median of {base_res.get('n_runs', 1)} run(s) per server · {active_key}")

metric_col1, metric_col2 = st.columns(2)

def show(label, value):
    st.markdown(f"**{label}**")
    st.markdown(f"### `{value}`")

with metric_col1:
    st.subheader("Baseline")
    st.markdown("---")
    if base_res:
        show("TTFT", f"{base_res['ttft']:.3f}s")
        show("TPOT (decode)", f"{base_res['tpot_ms']:.2f} ms/token")
        show("Decode speed", f"{base_res['decode_tps']:.1f} tok/s")
        show("End-to-End Latency", f"{base_res['latency']:.2f}s")
        st.caption(f"{int(base_res['tokens'])} output tokens · {int(base_res['prompt_tokens']):,} input tokens")
    else:
        st.markdown("`—`")

with metric_col2:
    st.subheader("Speculative")
    st.markdown("---")
    if spec_res and base_res:
        ttft_diff = (spec_res["ttft"] - base_res["ttft"]) / max(1e-6, base_res["ttft"]) * 100
        show("TTFT", f"{spec_res['ttft']:.3f}s")
        (st.warning if ttft_diff > 0 else st.success)(
            f"{'↑' if ttft_diff > 0 else '↓'} {abs(ttft_diff):.0f}% {'slower' if ttft_diff > 0 else 'faster'}")

        decode_speedup = base_res["tpot_ms"] / max(1e-6, spec_res["tpot_ms"])
        show("TPOT (decode)", f"{spec_res['tpot_ms']:.2f} ms/token")
        (st.success if decode_speedup >= 1 else st.warning)(f"Decode speedup: {decode_speedup:.2f}×")

        show("Decode speed", f"{spec_res['decode_tps']:.1f} tok/s")

        e2e_speedup = base_res["latency"] / max(1e-6, spec_res["latency"])
        show("End-to-End Latency", f"{spec_res['latency']:.2f}s")
        (st.info if e2e_speedup >= 1 else st.warning)(f"End-to-end speedup: {e2e_speedup:.2f}×")

        show("Draft Acceptance Rate", f"{spec_res.get('run_rate', 0):.1f}%")
        show("Mean Acceptance Length", f"{spec_res.get('mean_accept_len', 0):.2f} (max {NUM_SPEC_TOKENS + 1})")
        st.caption(f"{int(spec_res['tokens'])} output tokens")
    else:
        st.markdown("`—`")

if spec_res:
    with st.expander("▼ Advanced Speculative Metrics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted Draft Tokens", f"{int(spec_res.get('run_accepted', 0)):,}")
        c2.metric("Proposed Draft Tokens", f"{int(spec_res.get('run_drafted', 0)):,}")
        c3.metric("Verification Steps", f"{int(spec_res.get('run_steps', 0)):,}")
        st.caption("Counters are read from the speculative server's /metrics before and after each run. "
                   "Mean acceptance length = 1 + accepted ÷ steps.")

# Export everything measured so far
if st.session_state.results:
    rows = []
    for key, r in st.session_state.results.items():
        for side in ("base", "spec"):
            if r.get(side):
                row = {"experiment": key, "server": side}
                row.update({k: v for k, v in r[side].items() if k != "text"})
                rows.append(row)
    st.download_button("Download all results (CSV)", pd.DataFrame(rows).to_csv(index=False), "results.csv")
