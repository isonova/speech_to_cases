#!/usr/bin/env python3
"""
summarize_cases.py

Enhanced summarizer that merges the old fraud-aware logic with the new "Option A"
clean factual summaries. By default the script produces short, factual summaries
(Option A). Classification, flags and risk scoring are available behind the
`--classify` flag to preserve the original functionality.

Input: cases.json (format: {"cases": ["text1", "text2", ...]})
Output (default): summaries.json  (list of {case_index, text, summary})
If --classify is used, the output will include category, flags and risk_score
for each entry (backwards-compatible with the previous extended output).

Reference original project export:
/mnt/data/AIPRM-export-chatgpt-thread_Call-center-case-segmentation_2025-11-20T23_27_47.292Z.md
"""

import argparse
import json
import math
import re
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, pipeline

# ------------------- Utility & cleaning -------------------

def compute_safe_max_length(text: str, model_name: str = "sshleifer/distilbart-cnn-12-6", requested_max: int = None) -> int:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        enc = tokenizer(text, truncation=False)
        input_len = len(enc["input_ids"])
    except Exception:
        input_len = max(50, len(text.split()))
    suggested = max(20, int(input_len * 0.45))
    if requested_max:
        return min(requested_max, max(20, suggested))
    return min(200, suggested)


def clean_text_for_model(text: str) -> str:
    """Remove transcript markers, excessive whitespace and repeated fillers."""
    t = re.sub(r"={3,}.*?={3,}", "", text)
    t = re.sub(r"\s+", " ", t).strip()
    # collapse repeated filler words (e.g., "okay okay okay" -> "okay")
    t = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", t, flags=re.IGNORECASE)
    # remove sequences of short directional instructions that don't add semantic value
    t = re.sub(r"\b(click|tap|press|select|choose|open)\b(?:[^.?!]*?){1,3}(?:[.?!])", "", t, flags=re.I)
    t = t.strip()
    return t

# ------------------- Heuristics for classification (optional) -------------------
CATEGORY_KEYWORDS = {
    "Remote Access Attempt": ["anydesk", "teamviewer", "remote", "access code", "access id", "give me the numbers"],
    "App Install / Payment App": ["install", "download", "app", "cash app", "payment app", "qr code", "scan qr"],
    "Verification / Identity": ["verify", "verification", "manager", "confirm your", "identity", "id"],
    "Payment / Withdrawal Request": ["withdraw", "withdrawal", "transfer", "bank", "refund"],
    "Support / Legit Help": ["support", "help", "customer service", "finance department"],
}

FLAG_KEYWORDS = {
    "remote_access": ["anydesk", "teamviewer", "remote", "give me the numbers", "access code", "control your"],
    "requests_codes": [r"\b\d{3,}\b", "access code", "code", "pin"],
    "app_install": ["download", "install", "get the app"],
    "qr_scan": ["qr", "scan"],
    "payment_request": ["withdraw", "transfer", "send money", "refund"],
    "urgency": ["now", "immediately", "quick"],
}


def detect_flags_and_category(text: str) -> Dict[str, Any]:
    lower = text.lower()
    flags = {}
    score = 0.0
    for flag, kws in FLAG_KEYWORDS.items():
        found = any((re.search(kw, lower) if kw.startswith("\\") else kw in lower) for kw in kws)
        flags[flag] = bool(found)
    # category scoring
    cat_scores = {cat: sum(1 for kw in kws if kw in lower) for cat, kws in CATEGORY_KEYWORDS.items()}
    best_cat = max(cat_scores, key=lambda k: cat_scores[k])
    category = best_cat if cat_scores[best_cat] > 0 else "Other"
    if flags.get("remote_access"):
        score += 35
    if flags.get("app_install"):
        score += 20
    if flags.get("requests_codes"):
        score += 20
    if flags.get("payment_request"):
        score += 15
    if flags.get("qr_scan"):
        score += 10
    if flags.get("urgency"):
        score += 8
    if flags.get("remote_access") and "support" in lower:
        score += 5
    risk_score = int(min(100, math.floor(score)))
    return {"category": category, "flags": flags, "risk_score": risk_score}

# ------------------- Summarization -------------------

def make_summarizer(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    # prefer text2text/summarization model
    summarizer = pipeline("summarization", model=model_name, device=device)
    return summarizer


def abstractive_summary(summarizer, text: str, requested_max: int = None) -> str:
    safe_max = compute_safe_max_length(text, model_name=summarizer.model.name_or_path, requested_max=requested_max)
    chunk = text if len(text) < 3000 else text[:3000]
    try:
        out = summarizer(chunk, max_length=safe_max, min_length=8, do_sample=False)
        summary_text = out[0].get("summary_text", out[0].get("generated_text", "")).strip()
    except Exception:
        summary_text = " ".join(chunk.split()[:40]) + ("..." if len(chunk.split()) > 40 else "")
    summary_text = re.sub(r"\s+", " ", summary_text).strip()
    return summary_text

# ------------------- Orchestration -------------------

def process_cases(input_path: str, output_path: str, model_name: str = "google/flan-t5-base", max_len: int = None, classify: bool = False):
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Input JSON must contain a top-level 'cases' array.")
    print(f"[+] Loaded {len(cases)} cases from {input_path}")

    summarizer = make_summarizer(model_name)

    results = []
    for i, raw_text in enumerate(cases, start=1):
        text = clean_text_for_model(raw_text)
        if not text:
            summary = ""
            meta = {"category": "Empty", "flags": {}, "risk_score": 0}
        else:
            # Default behavior (Option A): short factual summary
            # For very short cleaned segments, return the cleaned text
            if len(text.split()) < 12:
                summary = text
            else:
                summary = abstractive_summary(summarizer, text, requested_max=max_len)
            # Optional classification
            meta = detect_flags_and_category(text) if classify else {"category": None, "flags": {}, "risk_score": 0}

        entry = {"case_index": i, "text": raw_text, "summary": summary}
        # attach classification fields only if requested (keeps output minimal by default)
        if classify:
            entry.update({"category": meta["category"], "flags": meta["flags"], "risk_score": meta["risk_score"]})

        results.append(entry)
        print(f"  - case {i}: summary_length={len(summary.split())}, classified={classify}")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"[+] Wrote {len(results)} entries to {output_path}")


# ------------------- CLI -------------------

def cli():
    p = argparse.ArgumentParser(description="Summarize cases (Option A by default). Add --classify to include categories/flags/risk.")
    p.add_argument("cases_file", help="Path to cases.json")
    p.add_argument("--out", default="summaries.json", help="Output file (default summaries.json)")
    p.add_argument("--model", default="google/flan-t5-base", help="Summarization model (default google/flan-t5-base)")
    p.add_argument("--max_len", type=int, default=None, help="Optional max length for summaries (tokens)")
    p.add_argument("--classify", action="store_true", help="Include category/flags/risk in output (optional)")
    args = p.parse_args()

    process_cases(args.cases_file, args.out, model_name=args.model, max_len=args.max_len, classify=args.classify)


if __name__ == "__main__":
    cli()
# ------------------- End of summarize_cases.py -------------------