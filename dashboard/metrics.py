import os
import requests


def get_metric_counter(metrics_text: str, metric_name: str) -> float:
    total = 0.0

    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue

        if line.startswith(metric_name):
            try:
                total += float(line.split()[-1])
            except ValueError:
                pass

    return total


def get_spec_counters():
    response = requests.get(
        os.environ["SPEC_METRICS_URL"],
        timeout=10,
    )

    response.raise_for_status()

    metrics = response.text

    return {
        "accepted_tokens": get_metric_counter(
            metrics,
            "vllm:spec_decode_num_accepted_tokens_total",
        ),
        "draft_tokens": get_metric_counter(
            metrics,
            "vllm:spec_decode_num_draft_tokens_total",
        ),
        "draft_cycles": get_metric_counter(
            metrics,
            "vllm:spec_decode_num_drafts",
        ),
    }


def compute_spec_delta(before, after):
    accepted_delta = (
        after["accepted_tokens"]
        - before["accepted_tokens"]
    )

    draft_delta = (
        after["draft_tokens"]
        - before["draft_tokens"]
    )

    draft_cycles_delta = (
        after["draft_cycles"]
        - before["draft_cycles"]
    )

    acceptance_rate = None

    if draft_delta > 0:
        acceptance_rate = (
            accepted_delta
            / draft_delta
        )

    mean_accepted_length = None

    if draft_cycles_delta > 0:
        mean_accepted_length = (
            1
            + (
                accepted_delta
                / draft_cycles_delta
            )
        )

    return {
        "accepted_delta": accepted_delta,
        "draft_delta": draft_delta,
        "draft_cycles_delta": draft_cycles_delta,
        "acceptance_rate": acceptance_rate,
        "mean_accepted_length": mean_accepted_length,
    }


def calculate_run_comparison(
    baseline,
    speculative,
):
    throughput_speedup = None

    if baseline["output_tps"] > 0:
        throughput_speedup = (
            speculative["output_tps"]
            / baseline["output_tps"]
        )

    latency_reduction = None

    if baseline["latency"] > 0:
        latency_reduction = (
            (
                baseline["latency"]
                - speculative["latency"]
            )
            / baseline["latency"]
        )

    return {
        "throughput_speedup": throughput_speedup,
        "latency_reduction": latency_reduction,
    }