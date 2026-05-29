import os
import streamlit as st
from client import run_inference
from metrics import get_spec_counters, compute_spec_delta, calculate_run_comparison


st.set_page_config(
    page_title="Speculative Decoding Demo",
    layout="wide",
)

st.title("Speculative Decoding with vLLM")

prompt = st.text_area(
    "Prompt",
    "Write a Python function that calculates Fibonacci numbers. Explain it briefly.",
)

max_tokens = st.slider("Max tokens", 64, 1024, 256, step=64)

if st.button("Run Comparison: baseline vs speculative"):

    spec_before = get_spec_counters()

    with st.spinner("Running baseline inference..."):
        baseline = run_inference(
            url=os.environ["BASELINE_URL"],
            model=os.environ["TARGET_MODEL"],
            prompt=prompt,
            max_tokens=max_tokens,
        )

    with st.spinner("Running speculative inference..."):
        speculative = run_infer(
            url=os.environ["SPEC_URL"],
            model=os.environ["TARGET_MODEL"],
            prompt=prompt,
            max_tokens=max_tokens,
        )

    spec_after = get_spec_counters()
    spec_stats = compute_spec_delta(spec_before, spec_after)
    comparison = calculate_run_comparison(baseline, speculative)

    st.divider()

    st.subheader(
        "Headline Results"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Throughput Speedup",
        (
            f"{comparison['throughput_speedup']:.2f}x"
            if comparison["throughput_speedup"]
            else "N/A"
        ),
    )

    col2.metric(
        "Latency Reduction",
        (
            f"{comparison['latency_reduction']*100:.1f}%"
            if comparison["latency_reduction"]
            is not None
            else "N/A"
        ),
    )

    col3.metric(
        "Acceptance Rate",
        (
            f"{spec_stats['acceptance_rate']*100:.1f}%"
            if spec_stats["acceptance_rate"]
            is not None
            else "N/A"
        ),
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Baseline")

        st.metric(
            "Latency",
            f"{baseline['latency']:.2f}s",
        )

        st.metric(
            "TTFT",
            (
                f"{baseline['ttft']:.2f}s"
                if baseline["ttft"]
                else "N/A"
            ),
        )

        st.metric(
            "Output Tokens", baseline["output_tokens"],)

        st.metric("Output Throughput", f"{baseline['output_tps']:.2f} tok/s")

        st.code(baseline["text"])

    with right:

        st.subheader(
            "Speculative"
        )

        st.metric(
            "Latency",
            f"{speculative['latency']:.2f}s",
        )

        st.metric(
            "TTFT",
            (
                f"{speculative['ttft']:.2f}s"
                if speculative["ttft"]
                else "N/A"
            ),
        )

        st.metric(
            "Output Tokens",
            speculative["output_tokens"],
        )

        st.metric(
            "Output Throughput",
            f"{speculative['output_tps']:.2f} tok/s",
        )

        st.metric(
            "Mean Accepted Length",
            (
                f"{spec_stats['mean_accepted_length']:.2f}"
                if spec_stats[
                    "mean_accepted_length"
                ]
                else "N/A"
            ),
        )

        st.code(
            speculative["text"]
        )

    st.divider()

    st.subheader(
        "Speculative Decoding Internals"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Draft Tokens Proposed",
        int(
            spec_stats["draft_delta"]
        ),
    )

    col2.metric(
        "Draft Tokens Accepted",
        int(
            spec_stats["accepted_delta"]
        ),
    )

    col3.metric(
        "Draft Cycles",
        int(
            spec_stats[
                "draft_cycles_delta"
            ]
        ),
    )