"""Batch runner for VoiceOps 2026 Step 3 multi-turn conversations.

Runs practice/conversations.csv grouped by conversation_id in turn order.
Outputs results to outputs/step3_results.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.data import DATASET_DIR
from src.conversation import ConversationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval.run_step3")


def run_conversations(
    conversations_csv: str | Path | None = None,
    output_csv: str | Path = "outputs/step3_results.csv",
) -> List[Dict[str, Any]]:
    """Execute all conversations turn by turn and save outputs."""
    csv_path = Path(conversations_csv) if conversations_csv else DATASET_DIR / "practice/conversations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {csv_path}")

    out_path = Path(output_csv)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    results: List[Dict[str, Any]] = []

    # Group by conversation_id in order of appearance
    conv_ids = df["conversation_id"].unique()
    logger.info("Starting run for %d conversations (%d total turns)", len(conv_ids), len(df))

    for conv_id in conv_ids:
        conv_rows = df[df["conversation_id"] == conv_id].copy()
        conv_rows["turn_num"] = conv_rows["turn"].astype(int)
        conv_rows = conv_rows.sort_values("turn_num")

        # Initialize fresh agent state for each conversation
        agent = ConversationAgent()

        for _, row in conv_rows.iterrows():
            customer_msg = row.get("customer_message", "")
            turn_res = agent.respond(customer_text=customer_msg)

            decision = turn_res.get("decision", "")
            reply = turn_res.get("reply", "")
            tool_calls = turn_res.get("tool_calls", [])
            used_sections = turn_res.get("used_sections", [])
            dropped_sections = turn_res.get("dropped_sections", [])
            state_after = turn_res.get("state", {})
            dropped_sections_count = state_after.get("dropped_sections_count", len(dropped_sections))

            result_entry = {
                "conversation_id": conv_id,
                "turn": row.get("turn", ""),
                "scenario": row.get("scenario", ""),
                "language": row.get("language", ""),
                "order_id": row.get("order_id", ""),
                "customer_id": row.get("customer_id", ""),
                "customer_message": customer_msg,
                "decision": decision,
                "reply": reply,
                "tool_calls": json.dumps(tool_calls, ensure_ascii=False),
                "used_sections": " ; ".join(used_sections),
                "dropped_sections_count": dropped_sections_count,
                "state_after": json.dumps(state_after, ensure_ascii=False),
            }
            results.append(result_entry)

    fieldnames = [
        "conversation_id",
        "turn",
        "scenario",
        "language",
        "order_id",
        "customer_id",
        "customer_message",
        "decision",
        "reply",
        "tool_calls",
        "used_sections",
        "dropped_sections_count",
        "state_after",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info("Saved %d turn results to %s", len(results), out_path)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 3 multi-turn conversations")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to practice/conversations.csv (default: DATASET_DIR/practice/conversations.csv)",
    )
    parser.add_argument(
        "--output",
        default="outputs/step3_results.csv",
        help="Path to output CSV (default: outputs/step3_results.csv)",
    )
    args = parser.parse_args()

    run_conversations(conversations_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
