import time
import requests
import streamlit as st

BASELINE_URL = "http://localhost:8000/v1/chat/completions"
SPECULATIVE_URL = "http://localhost:8001/v1/chat/completions"

st.set_page_config(
    page_title="vLLM Speculative Decoding Demo",
    layout="wide",
)

st.title("vLLM Baseline vs Speculative Decoding")

prompt = st.text_area(
    "Prompt",
    placeholder="Write your prompt here...",
    height=140,
)

max_tokens = st.slider("Max tokens", 50, 1000, 300, step=50)
temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)


def run_inference(url: str, prompt: str) -> dict:
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=300)
    end = time.perf_counter()

    response.raise_for_status()
    data = response.json() #Convert HTTP JSON response into dict.

    text = data["choices"][0]["message"]["content"] #Extract the generated text from the response.
    usage = data.get("usage", {}) #Get the usage information from the response, which includes token counts.

    # Calculate performance metrics based on the response and timing information.
    completion_tokens = usage.get("completion_tokens", 0)
    total_latency = end - start
    tokens_per_second = (
        completion_tokens / total_latency if total_latency > 0 else 0
    )

    return {
        "text": text,
        "latency": total_latency,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
    }


col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline")
    if st.button("Run Baseline"):
        with st.spinner("Running baseline inference..."):
            result = run_inference(BASELINE_URL, prompt)

        st.metric("Total latency", f"{result['latency']:.2f}s")
        st.metric("Generated tokens", result["completion_tokens"])
        st.metric("Tokens/sec", f"{result['tokens_per_second']:.2f}")
        st.code(result["text"], language="text")

with col2:
    st.subheader("Speculative")
    if st.button("Run Speculative"):
        with st.spinner("Running speculative inference..."):
            result = run_inference(SPECULATIVE_URL, prompt)

        st.metric("Total latency", f"{result['latency']:.2f}s")
        st.metric("Generated tokens", result["completion_tokens"])
        st.metric("Tokens/sec", f"{result['tokens_per_second']:.2f}")
        st.code(result["text"], language="text")