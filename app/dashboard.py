# app.py
import time
import requests
import streamlit as st
from openai import OpenAI

BASELINE_URL = "http://127.0.0.1:8000/v1"
SPEC_URL = "http://127.0.0.1:8001/v1"

TARGET_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

baseline_client = OpenAI(base_url=BASELINE_URL, api_key="EMPTY")
spec_client = OpenAI(base_url=SPEC_URL, api_key="EMPTY")

PROMPTS = {
    "Simple explanation": "Explain speculative decoding in 5 beginner-friendly bullet points.",
    "Long generation": "Write a 500-word explanation of how vLLM serves requests, including batching, KV cache, and token generation.",
    "Harder reasoning": "Given a Python service that gets slower as concurrent requests increase, explain three possible bottlenecks and how you would debug them."
}

st.set_page_config(page_title="vLLM Speculative Decoding Demo", layout="wide")

st.title("vLLM Baseline vs Speculative Decoding")

preset = st.radio(
    "Choose prompt type",
    list(PROMPTS.keys()),
    horizontal=True
)

prompt = st.text_area("Prompt", value=PROMPTS[preset], height=160)

max_tokens = st.slider("Max tokens", 64, 1024, 512, step=64)
temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

def run_request(client, model, prompt):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    end = time.perf_counter()
    latency = end - start

    text = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens if response.usage else len(text.split())
    throughput = output_tokens / latency if latency > 0 else 0

    return {
        "text": text,
        "latency": latency,
        "output_tokens": output_tokens,
        "throughput": throughput,
    }

if st.button("Run baseline and speculative", type="primary"):
    with st.spinner("Running baseline..."):
        baseline = run_request(baseline_client, TARGET_MODEL, prompt)

    with st.spinner("Running speculative..."):
        speculative = run_request(spec_client, TARGET_MODEL, prompt)

    speedup = baseline["latency"] / speculative["latency"]

    st.subheader("Comparison")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline latency", f"{baseline['latency']:.2f}s")
    c2.metric("Speculative latency", f"{speculative['latency']:.2f}s")
    c3.metric("Speedup", f"{speedup:.2f}x")
    c4.metric("Spec throughput", f"{speculative['throughput']:.1f} tok/s")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Baseline tokens/sec", f"{baseline['throughput']:.1f}")
    c6.metric("Spec tokens/sec", f"{speculative['throughput']:.1f}")
    c7.metric("Baseline output tokens", baseline["output_tokens"])
    c8.metric("Spec output tokens", speculative["output_tokens"])

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Baseline output")
        st.write(baseline["text"])

    with right:
        st.subheader("Speculative output")
        st.write(speculative["text"])