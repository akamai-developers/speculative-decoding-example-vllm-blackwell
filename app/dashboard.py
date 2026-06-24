import streamlit as st
import asyncio
import httpx
import time

# Page configuration for a widescreen presentation view
st.set_page_config(page_title="vLLM Speculative Decoding Showdown", layout="wide")

st.title("🚀 vLLM Speculative Decoding Showdown")
st.subheader("NVIDIA Blackwell Live Profiling Demo")

TARGET_MODEL = os.environ["TARGET_MODEL"]
BASELINE_URL = "http://127.0.0.1:8000/v1/chat/completions"
SPECULATIVE_URL = "http://127.0.0.1:8001/v1/chat/completions"

# -------------------------------------------------------------------
# Async streaming worker for vLLM endpoints
# -------------------------------------------------------------------
async def stream_from_vllm(url, prompt, placeholder, metric_placeholder):
    payload = {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": True
    }
    
    text_buffer = ""
    token_count = 0
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    placeholder.error(f"Error: Server returned status {response.status_code}")
                    return
                    
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        # Parsing JSON chunk manually for lightweight execution
                        try:
                            import json
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                text_buffer += delta
                                token_count += 1
                                # Dynamic UI streaming text update
                                placeholder.markdown(text_buffer + "▌")
                        except Exception:
                            pass
    except Exception as e:
        placeholder.error(f"Connection Failed: {str(e)}")
        return

    end_time = time.time()
    total_time = end_time - start_time
    tpot = (total_time / token_count) * 1000 if token_count > 0 else 0
    tok_per_sec = token_count / total_time if total_time > 0 else 0
    
    # Strip the cursor block once finished
    placeholder.markdown(text_buffer)
    
    # Render final telemetry numbers immediately on UI
    metric_placeholder.markdown(f"""
    **📊 Inference Summary:**
    * Total Time: `{total_time:.2f}s`
    * Throughput: `{tok_per_sec:.1f} tok/s`
    * Avg TPOT: `{tpot:.1f} ms/tok`
    """)

# Helper to bundle parallel async execution tasks
async def run_simultaneous_showdown(prompt, col1_text, col1_metrics, col2_text, col2_metrics):
    await asyncio.gather(
        stream_from_vllm(BASELINE_URL, prompt, col1_text, col1_metrics),
        stream_from_vllm(SPECULATIVE_URL, prompt, col2_text, col2_metrics)
    )

# -------------------------------------------------------------------
# Streamlit Layout Construction
# -------------------------------------------------------------------
prompt_input = st.text_area(
    "Enter a prompt to dispatch to both inference nodes:",
    value="Write a clean Python implementation of a quicksort algorithm with comments.",
    height=100
)

# Preset scenario triggers for quick switching on stage
st.markdown("**Quick Preset Prompts for the Stage:**")
col_p1, col_p2 = st.columns(2)
if col_p1.button("🎯 High Acceptance Case (Code Generation)"):
    st.session_state.prompt_input = "Write a comprehensive Python script to calculate Fibonacci numbers sequentially."
if col_p2.button("🌀 Low Acceptance Case (Chaotic/Creative Writing)"):
    st.session_state.prompt_input = "Write a highly abstract surrealist poem where every alternative word switches between baking vocabulary and quantum mechanics principles."

# Main action trigger
if st.button("🔥 Run Live Showdown", type="primary"):
    # Split UI screen evenly into baseline and speculative sectors
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Standard Baseline Node (Port 8000)")
        st.caption("Running pure unquantized Llama 3.1 8B Instruct")
        baseline_text = st.empty()
        baseline_metrics = st.empty()
        baseline_text.info("Awaiting execution sync...")
        
    with col2:
        st.header("Speculative Target Node (Port 8001)")
        st.caption("Accelerated via Llama 3.2 3B Instruct Draft Execution")
        spec_text = st.empty()
        spec_metrics = st.empty()
        spec_text.info("Awaiting execution sync...")
        
    # Kick off the underlying asynchronous event loop
    asyncio.run(run_simultaneous_showdown(
        prompt_input, 
        baseline_text, baseline_metrics, 
        spec_text, spec_metrics
    ))