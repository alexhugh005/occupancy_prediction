"""Replay rows from datatest2.txt to the occupancy API, one request per row,
as if each row were a live sensor event.

Usage:
    python replay_events.py datatest2.txt --limit 20
    python replay_events.py datatest2.txt --delay 1.0        # ~1 event/second
    python replay_events.py datatest2.txt --url http://localhost:8000 --route /predict

The script reads /openapi.json from your running service, finds the POST route
and its JSON fields, and maps CSV columns to those fields by name. If a field
can't be matched, it tells you, and you can edit ALIASES below.
"""
import argparse
import re
import sys
import time

import pandas as pd
import requests

# request-field name (normalized) -> CSV column name (normalized)
ALIASES = {
    "timestamp": "date",
    "datetime": "date",
    "time": "date",
    "ts": "date",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_rows(path: str) -> pd.DataFrame:
    # The header has 7 names but each row has 8 fields (a leading row number),
    # so use column 0 as the index; "date" then labels the timestamp column.
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def discover_route(base: str, route: str | None):
    spec = requests.get(f"{base}/openapi.json", timeout=5).json()
    paths = spec["paths"]
    candidates = [route] if route else [p for p, ops in paths.items() if "post" in ops]
    if not candidates or candidates[0] not in paths:
        sys.exit(f"No POST route found. Routes available: {list(paths)}")
    path = candidates[0]
    schema = paths[path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
    props = schema.get("properties")
    if not props:
        sys.exit(
            f"{path} doesn't take a flat JSON object (maybe a list/batch). "
            "Check http://localhost:8000/docs and adapt build_payload()."
        )
    return path, list(props)


def build_payload(row: pd.Series, fields: list[str]) -> dict:
    by_norm = {norm(c): c for c in row.index}
    payload, missing = {}, []
    for f in fields:
        key = ALIASES.get(norm(f), norm(f))
        col = by_norm.get(key)
        if col is None:
            missing.append(f)
            continue
        val = row[col]
        payload[f] = val.isoformat() if isinstance(val, pd.Timestamp) else float(val)
    if missing:
        print(f"  (no CSV column for field(s): {missing}; sent without them)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--route", default=None, help="e.g. /predict (auto-detected if omitted)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between events")
    args = ap.parse_args()

    df = load_rows(args.csv)
    route, fields = discover_route(args.url, args.route)
    print(f"POST {args.url}{route} with fields {fields}\n")

    for i, row in df.head(args.limit).iterrows():
        payload = build_payload(row, fields)
        r = requests.post(f"{args.url}{route}", json=payload, timeout=10)
        print(f"[{i}] {row['date']}  actual={int(row['Occupancy'])}  "
              f"-> {r.status_code} {r.text[:200]}")
        if args.delay:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
