#!/usr/bin/env python3
"""Smoke-test a local or tunneled LoMTL API using a real repository example."""

import argparse
import json
import sys
from pathlib import Path

import requests

DIMENSIONS = {
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
}
LABELS = {"Yes", "To some extent", "No"}
DEFAULT_SAMPLE = Path(__file__).resolve().parents[1] / "tests/sample_request.json"


def require_ok(response, endpoint):
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} returned HTTP {response.status_code}: {response.text}")
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/test_api.py --base-url http://127.0.0.1:8000
  python scripts/test_api.py --base-url https://random-name.trycloudflare.com

By default, the request is loaded from tests/sample_request.json. It is adapted
from the first conversation and GPT4 tutor response in
assets/data/test_data/test_sample.json.
""",
    )
    parser.add_argument("--base-url", required=True, help="API root, local or trycloudflare.com")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    with args.sample.open(encoding="utf-8") as sample_file:
        payload = json.load(sample_file)

    print(f"Testing API: {base_url}")
    print(f"Request sample: {args.sample}")
    print("Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    health = require_ok(requests.get(f"{base_url}/health", timeout=args.timeout), "/health")
    if health.get("model_ready") is not True:
        raise RuntimeError(f"Model is not ready: {health}")

    result = require_ok(
        requests.post(f"{base_url}/evaluate", json=payload, timeout=args.timeout),
        "/evaluate",
    )
    if set(result) != DIMENSIONS:
        raise RuntimeError(f"Expected exactly {sorted(DIMENSIONS)}, received {sorted(result)}")
    invalid = {task: label for task, label in result.items() if label not in LABELS}
    if invalid:
        raise RuntimeError(f"Invalid LoMTL labels: {invalid}")

    print("Health response:")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    print("\nEvaluation response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nAPI smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, requests.RequestException, RuntimeError) as exc:
        print(f"API smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
