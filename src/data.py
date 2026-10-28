"""Data loader and helper utilities for ZentroMart Capstone."""

from __future__ import annotations
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from repo root .env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

REFERENCE_DATE = "2026-10-20"
REFERENCE_DATE_OBJ = date(2026, 10, 20)

DATASET_DIR_ENV = os.getenv("DATASET_DIR")
if not DATASET_DIR_ENV:
    raise RuntimeError(
        "DATASET_DIR is not set. Set it in .env to the extracted "
        "Capstone_Project_Dataset_FTTT_2026 directory."
    )
DATASET_DIR = Path(DATASET_DIR_ENV)
if not DATASET_DIR.exists():
    raise FileNotFoundError(f"DATASET_DIR does not exist: {DATASET_DIR}")

def load_csv(rel_path: str) -> pd.DataFrame:
    """Load a CSV file with all columns as text (str) and preserving empty strings."""
    full_path = DATASET_DIR / rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"CSV file not found: {full_path}")
    return pd.read_csv(full_path, dtype=str, keep_default_na=False)

# Load all 9 CSVs
CUSTOMERS_DF = load_csv("data/customers.csv")
ORDERS_DF = load_csv("data/orders.csv")
ORDER_ITEMS_DF = load_csv("data/order_items.csv")
PRODUCTS_DF = load_csv("data/products.csv")
PINCODES_DF = load_csv("data/pincodes.csv")
HOLIDAYS_DF = load_csv("data/holidays.csv")
SUPPORT_LOGS_DF = load_csv("data/support_logs.csv")
QUESTIONS_DF = load_csv("practice/questions.csv")
CONVERSATIONS_DF = load_csv("practice/conversations.csv")

# Set of holiday dates for fast lookup
HOLIDAY_DATES = set(HOLIDAYS_DF["date"].tolist())

def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a customer row as a dictionary by customer_id."""
    cid = str(customer_id).strip()
    match = CUSTOMERS_DF[CUSTOMERS_DF["customer_id"] == cid]
    if match.empty:
        return None
    return match.iloc[0].to_dict()

def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the order row plus its items."""
    oid = ''.join(ch for ch in str(order_id) if '0' <= ch <= '9')
    match = ORDERS_DF[ORDERS_DF["order_id"] == oid]
    if match.empty:
        return None
    order_dict = match.iloc[0].to_dict()
    items_match = ORDER_ITEMS_DF[ORDER_ITEMS_DF["order_id"] == oid]
    order_dict["items"] = items_match.to_dict(orient="records")
    return order_dict

def get_product(sku: str) -> Optional[Dict[str, Any]]:
    """Retrieve a product row as a dictionary by sku."""
    s = str(sku).strip()
    match = PRODUCTS_DF[PRODUCTS_DF["sku"] == s]
    if match.empty:
        return None
    return match.iloc[0].to_dict()

def _parse_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    d_str = str(d).strip()
    if "T" in d_str:
        return datetime.fromisoformat(d_str).date()
    return date.fromisoformat(d_str[:10])

def business_days_between(start: Union[str, date, datetime],
                          end: Union[str, date, datetime],
                          inclusive: bool = False) -> int:
    """Calculate business days between start and end.
    
    A business day is Monday to Saturday (Sundays excluded: weekday == 6),
    skipping warehouse holidays in data/holidays.csv.
    
    By default (inclusive=False), counts elapsed business days in (start, end].
    If inclusive=True, counts business days in [start, end].
    """
    d_start = _parse_date(start)
    d_end = _parse_date(end)
    
    if d_start > d_end:
        # Negative business days if start is after end
        return -business_days_between(d_end, d_start, inclusive=inclusive)
    
    current = d_start if inclusive else d_start + timedelta(days=1)
    business_days = 0
    
    while current <= d_end:
        # 0 = Monday, ..., 5 = Saturday, 6 = Sunday
        if current.weekday() != 6:
            iso_date = current.isoformat()
            if iso_date not in HOLIDAY_DATES:
                business_days += 1
        current += timedelta(days=1)
        
    return business_days
