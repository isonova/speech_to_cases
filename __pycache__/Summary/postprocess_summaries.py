#!/usr/bin/env python3
"""
Post-process pipeline_output.json / summaries.json:
- remove URLs
- collapse repeated instructions
- convert instruction-heavy blocks into short intent statements
- trim length to a concise factual summary (Option A)
Produces:
 - pipeline_output_clean.json
 - pipeline_output_clean.csv
 - pipeline_output_clean.xlsx (if pandas available)
"""

import json, re, argparse, csv
from pathlib import Path

# simple intent mapping from keywords -> short factual sentence
INTENT_RULES = [
    (r"\banydesk\b|\bany desk\b|\bteamviewer\b", "Agent asks user to install a remote-access app (AnyDesk/TeamViewer) and provide the connection code."),
    (r"\bdownload\b|\binstall\b|\bget the app\b", "Agent instructs user to download/install a mobile app."),
    (r"\bqr\b|\bscan\b|\bcamera\b", "Agent instructs user to scan a QR code or use the camera."),
    (r"\bwithdraw\b|\bwithdrawal\b|\btransfer\b|\bbank\b|\bmoney back\b|\brefund\b", "Agent discusses returning/withdrawing funds or bank transfer steps."),
    (r"\bmanager\b|\bverify\b|\bidentity\b|\bconfirm your\b", "Agent requests verification or mentions account/manager details."),
    (r"\bfull access\b|\bpermissions\b|\bcontrol your\b|\bcontrol your phone\b", "Agent requests broad permissions / full access to device."),
    (r"\bcode\b|\baccess code\b|\baccess id\b|\b\d{3,}\b", "Agent requests numeric access codes or connection IDs."),
]

# patterns to remove/harden
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+\b", flags=re.I)
MULTI_WHITESPACE = re.compile(r"\s+")
REPEATED_PHRASE = re.compile(r"\b(\w+(?:\s+\w+){0,4})(?:\s+\1){1,}", flags=re.I)  # collapse small repeated phrases

def clean_text(s: str) -> str:
    if not s: 
        return ""
    t = s
    t = re.sub(r"={3,}.*?={3,}", "", t)
    t = URL_RE.sub("", t)
    t = REPEATED_PHRASE.sub(r"\1", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\s*([.,!?])\s*", r"\1 ", t).strip()
    t = re.sub(r"[.]{2,}", ".", t)
    return t.strip()

def detect_intent(cleaned_text: str) -> str:
    lower = cleaned_text.lower()
    for pattern, intent_sent in INTENT_RULES:
        if re.search(pattern, lower):
            return intent_sent
    return None

def shorten_to_sentence(text: str, max_words=25):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "."

def postprocess_entry(entry: dict) -> dict:
    raw = entry.get("summary") or entry.get("text") or ""
    cleaned = clean_text(raw)
    intent = detect_intent(cleaned)

    if intent:
        if re.search(r"\b(click|open|install|download|scan|press|accept|start|give)\b", cleaned, flags=re.I) or len(cleaned.split()) > 20:
            final = intent
        else:
            final = cleaned
    else:
        final = cleaned

    final = re.sub(r"(\. ){2,}", ". ", final)
    final = MULTI_WHITESPACE.sub(" ", final).strip()

    if not final:
        txt = clean_text(entry.get("text", ""))
        final = shorten_to_sentence(txt, max_words=30)

    final = shorten_to_sentence(final, max_words=25)
    final = final.strip()
    final = re.sub(r"[^A-Za-z0-9\.\,\?\! ]+$", "", final).strip()

    out = dict(entry)
    out["summary_clean"] = final
    return out

def main(input_path: str, output_json: str, output_csv: str, output_xlsx: str=None):
    p = Path(input_path)
    if not p.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    data = json.loads(p.read_text(encoding="utf-8"))

    processed = [postprocess_entry(item) for item in data]

    # JSON
    Path(output_json).write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote cleaned JSON -> {output_json}")

    # CSV – FIXED QUOTES HERE
    fieldnames = ["case_index", "summary_clean", "summary", "text"]
    sample = processed[0] if processed else {}
    if "category" in sample:
        fieldnames += ["category", "risk_score", "flags"]

    with open(output_csv, "w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in processed:
            row = {k: r.get(k,"") for k in fieldnames}
            if "flags" in r and isinstance(r["flags"], dict):
                row["flags"] = json.dumps(r["flags"], ensure_ascii=False)
            writer.writerow(row)
    print(f"Wrote cleaned CSV -> {output_csv}")

    # XLSX (optional)
    if output_xlsx:
        try:
            import pandas as pd
            df = pd.DataFrame(processed)
            cols = fieldnames + [c for c in df.columns if c not in fieldnames]
            df = df.reindex(columns=cols)
            df.to_excel(output_xlsx, index=False)
            print(f"Wrote cleaned XLSX -> {output_xlsx}")
        except Exception as e:
            print(f"Skipping XLSX (pandas not available or error): {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Postprocess pipeline_output.json to clean summaries.")
    ap.add_argument("--in", dest="input_path", default="pipeline_output.json", help="Input pipeline_output.json or summaries.json")
    ap.add_argument("--out-json", dest="out_json", default="pipeline_output_clean.json")
    ap.add_argument("--out-csv", dest="out_csv", default="pipeline_output_clean.csv")
    ap.add_argument("--out-xlsx", dest="out_xlsx", default="pipeline_output_clean.xlsx")
    args = ap.parse_args()
    main(args.input_path, args.out_json, args.out_csv, args.out_xlsx)
