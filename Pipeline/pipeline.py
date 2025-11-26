from pathlib import Path
import sys
import json
import os

# ---------------------------
# Make project root importable
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------
# Imports from your folders
# ---------------------------
from ASR.transcribe_call import transcribe_audio
from Segment.segment_cases_ml import segment_transcript
from Summary.summarize_cases import process_cases  # improved summarizer (Option A default)

# local path to the original uploaded project export (kept as metadata)
SOURCE_EXPORT_PATH = "/mnt/data/AIPRM-export-chatgpt-thread_Call-center-case-segmentation_2025-11-20T23_27_47.292Z.md"


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    """
    Lightweight CSV writer that does not require pandas.
    rows: list of dict
    fieldnames: list of columns in order
    """
    import csv
    with open(path, "w", encoding="utf-8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_xlsx(path, rows, fieldnames):
    """
    Try to write XLSX using pandas (preferred). If pandas not available, raise ImportError.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas required for XLSX output") from e

    # create DataFrame with consistent column order
    df = pd.DataFrame(rows)
    # ensure all expected cols exist
    for col in fieldnames:
        if col not in df.columns:
            df[col] = None
    df = df[fieldnames]
    # write excel (openpyxl engine used by default)
    df.to_excel(path, index=False)


def run_pipeline(audio_path,
                 merge_min_words=6,
                 smooth_window=3,
                 sim_threshold=0.28,
                 min_segment_words=35,
                 enable_classification=False,
                 summarizer_model="sshleifer/distilbart-cnn-12-6"):
    """
    Full pipeline:
    1) ASR -> transcript
    2) Segmentation -> cases.json
    3) Summarization -> summaries.json
    4) Compose outputs -> pipeline_output.json / .csv / .xlsx
    """

    # 1) ASR
    print("Running ASR...")
    transcript = transcribe_audio(audio_path)

    # 2) Segmentation
    print("Segmenting transcript...")
    segments = segment_transcript(
        transcript,
        merge_min_words=merge_min_words,
        smooth_window=smooth_window,
        sim_threshold=sim_threshold,
        min_segment_words=min_segment_words
    )

    # Save intermediate cases.json
    cases_obj = {"cases": segments}
    write_json("cases.json", cases_obj)
    print(f"Saved {len(segments)} segments to cases.json")

    # 3) Summarization using Summary.summarize_cases.process_cases
    # NOTE: your summarize_cases.py should accept 'classify=' for classification toggle
    print("Summarizing segments...")
    process_cases(
        "cases.json",
        "summaries.json",
        model_name=summarizer_model,
        max_len=None,
        classify=enable_classification,
    )

    # 4) Load summaries and combine into pipeline output
    with open("summaries.json", "r", encoding="utf-8") as fh:
        summary_entries = json.load(fh)

    pipeline_out = []
    for entry in summary_entries:
        item = {
            "case_index": entry.get("case_index"),
            "text": entry.get("text", ""),
            "summary": entry.get("summary", "")
        }
        # attach classification metadata if present
        if enable_classification:
            item["category"] = entry.get("category")
            item["flags"] = entry.get("flags")
            item["risk_score"] = entry.get("risk_score")
        # attach source export metadata for traceability
        item["source_export"] = SOURCE_EXPORT_PATH
        pipeline_out.append(item)

    # 5) Write JSON
    write_json("pipeline_output.json", pipeline_out)
    print("Saved pipeline_output.json")

    # 6) Write CSV (flatten non-scalar fields like flags into JSON strings)
    # Build rows with stable columns
    csv_fieldnames = ["case_index", "text", "summary"]
    if enable_classification:
        csv_fieldnames += ["category", "flags", "risk_score"]
    csv_fieldnames += ["source_export"]

    # prepare rows for CSV (stringify complex types)
    import json as _json
    csv_rows = []
    for r in pipeline_out:
        row = {
            "case_index": r.get("case_index"),
            "text": r.get("text", ""),
            "summary": r.get("summary", ""),
            "source_export": r.get("source_export", "")
        }
        if enable_classification:
            row["category"] = r.get("category", "")
            row["flags"] = _json.dumps(r.get("flags", {}), ensure_ascii=False)
            row["risk_score"] = r.get("risk_score", 0)
        csv_rows.append(row)

    write_csv("pipeline_output.csv", csv_rows, csv_fieldnames)
    print("Saved pipeline_output.csv")

    # 7) Try XLSX (best-effort)
    try:
        write_xlsx("pipeline_output.xlsx", csv_rows, csv_fieldnames)
        print("Saved pipeline_output.xlsx")
    except ImportError:
        print("pandas not available in environment — skipping XLSX. "
              "To enable XLSX, install pandas and openpyxl.")
    except Exception as e:
        print(f"Failed to write XLSX: {e}")

    return pipeline_out


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py sample_call.wav")
        sys.exit(1)
    audio_file = sys.argv[1]
    # You can toggle enable_classification=True to include category/flags/risk in CSV/XLSX
    run_pipeline(audio_file)
