"""Pre-download the configured model + tokenizer into the local cache.

Run during environment setup so the first server boot does not pay a cold
download cost. Idempotent: Hugging Face reuses the cached files on repeat runs.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    model = os.environ.get("FLUX_MODEL", "distilgpt2")
    print(f"Fetching tokenizer and weights for '{model}'...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)
    print(f"Model '{model}' is cached and ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
