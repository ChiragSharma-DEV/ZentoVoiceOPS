"""Order tools wrapper for VoiceOps 2026 Step 3.

Wraps the mock tools provided in DATASET_DIR/tools/tools.py without modifying the dataset.
Adds DATASET_DIR to sys.path to import lookup_order and create_ticket.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import DATASET_DIR from src.data
from src.data import DATASET_DIR

# Ensure DATASET_DIR is present on sys.path
_dataset_path_str = str(DATASET_DIR)
if _dataset_path_str not in sys.path:
    sys.path.insert(0, _dataset_path_str)

try:
    from tools.tools import lookup_order as _raw_lookup_order
    from tools.tools import create_ticket as _raw_create_ticket
except ImportError as exc:
    raise ImportError(
        f"Could not import tools from DATASET_DIR ({_dataset_path_str}): {exc}"
    ) from exc


def lookup_order_unverified(order_id: str) -> Dict[str, Any]:
    """Call mock lookup_order with order_id and no verification value.

    Returns verification_required message and no private order details.
    """
    clean_id = "".join(ch for ch in str(order_id) if ch.isdigit())
    return _raw_lookup_order(clean_id)


def lookup_order_verified(order_id: str, value: str) -> Dict[str, Any]:
    """Call mock lookup_order with order_id and caller-provided verification value.

    Passes only the digits the customer actually typed.
    Returns full order details if verified is True, else verified is False with no order details.
    """
    clean_id = "".join(ch for ch in str(order_id) if ch.isdigit())
    clean_val = "".join(ch for ch in str(value) if ch.isdigit())
    return _raw_lookup_order(clean_id, verification_value=clean_val)


def handover(
    summary: str,
    order_id: Optional[str] = None,
    reason: Optional[str] = None,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a human-handover ticket via create_ticket.

    Returns dict containing:
        created (bool): True if created, False if validation failed.
        ticket_id (str, optional): The generated ticket ID if created is True.
        error (str, optional): Validation error message if created is False.
    """
    clean_id = "".join(ch for ch in str(order_id) if ch.isdigit()) if order_id else None
    result = _raw_create_ticket(
        summary=summary,
        order_id=clean_id,
        reason=reason,
        intent=intent,
    )
    return result
