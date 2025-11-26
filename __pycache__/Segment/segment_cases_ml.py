# segment_cases_ml.py
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import sys

# ------- Helpers -------
def split_into_sentences(text: str):
    # naive but practical for transcripts
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def merge_short_sentences(sentences, min_words=6):
    merged = []
    buffer = ""
    for s in sentences:
        if len(s.split()) < min_words:
            buffer = (buffer + " " + s).strip()
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(s)
    if buffer:
        merged.append(buffer.strip())
    return merged

def batch_encode(embedder, sentences, batch_size=32):
    embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb = embedder.encode(batch, show_progress_bar=False)
        embs.append(emb)
    return np.vstack(embs)

def cosine_similarities(embeddings):
    # pairwise consecutive cosine similarities
    sims = []
    for i in range(1, len(embeddings)):
        a = embeddings[i-1]
        b = embeddings[i]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        sim = 0.0 if denom == 0 else float(np.dot(a, b) / denom)
        sims.append(sim)
    return np.array(sims)

def smooth(array, window=3):
    if window <= 1:
        return array
    pad = window // 2
    padded = np.pad(array, (pad, pad), mode='edge')
    sm = np.convolve(padded, np.ones(window)/window, mode='valid')
    return sm

def find_boundaries(smoothed_sims, threshold=0.28):
    # boundary at positions where sim drops below threshold
    # returns indices i meaning boundary before sentence i+1
    lows = np.where(smoothed_sims < threshold)[0]
    # filter adjacent duplicates (keep only when gap > 0)
    if len(lows) == 0:
        return []
    boundaries = [int(lows[0])]
    for idx in lows[1:]:
        if idx - boundaries[-1] > 0:
            boundaries.append(int(idx))
    return boundaries

def enforce_min_segment_length(sentences, boundaries, min_words=30):
    """
    Merge segments that are too short (in word-count) into neighbor.
    boundaries are indices in sims (i.e. between sentence i and i+1).
    """
    # build segments as list of (start_idx, end_idx inclusive)
    segs = []
    start = 0
    for b in boundaries:
        end = b  # boundary before sentence b+1 => include sentence b
        segs.append((start, end))
        start = b+1
    segs.append((start, len(sentences)-1))

    # compute word counts
    def seg_word_count(seg):
        s,e = seg
        return sum(len(sentences[i].split()) for i in range(s, e+1))

    i = 0
    while i < len(segs):
        if seg_word_count(segs[i]) < min_words:
            # merge with previous if exists else next
            if i > 0:
                # merge into previous
                s_prev, e_prev = segs[i-1]
                s_cur, e_cur = segs[i]
                segs[i-1] = (s_prev, e_cur)
                segs.pop(i)
                i = max(i-1, 0)
            elif i+1 < len(segs):
                # merge into next
                s_cur, e_cur = segs[i]
                s_next, e_next = segs[i+1]
                segs[i] = (s_cur, e_next)
                segs.pop(i+1)
            else:
                # single tiny segment, nothing to merge
                i += 1
        else:
            i += 1
    return segs

# ------- Main segmentation function -------
def segment_transcript(transcript,
                       merge_min_words=6,
                       smooth_window=3,
                       sim_threshold=0.28,
                       min_segment_words=35,
                       embed_model="all-MiniLM-L6-v2"):
    sentences = split_into_sentences(transcript)
    if len(sentences) == 0:
        return []

    # Pre-merge very short utterances
    sentences = merge_short_sentences(sentences, min_words=merge_min_words)

    # If very few sentences, return single segment
    if len(sentences) <= 4:
        return [" ".join(sentences)]

    # embeddings
    embedder = SentenceTransformer(embed_model)
    embeddings = batch_encode(embedder, sentences, batch_size=64)

    sims = cosine_similarities(embeddings)  # len = n_sentences-1
    smoothed = smooth(sims, window=smooth_window)

    raw_boundaries = find_boundaries(smoothed, threshold=sim_threshold)
    # convert boundary indices in sims to sentence boundary indexes:
    # sims index i => boundary before sentence i+1
    boundaries = raw_boundaries

    # enforce min segment word count by merging tiny segments
    seg_ranges = enforce_min_segment_length(sentences, boundaries, min_words=min_segment_words)

    # build text for each segment
    segments = []
    for s,e in seg_ranges:
        txt = " ".join(sentences[s:e+1])
        segments.append(txt.strip())

    return segments

# ------- CLI -------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment_cases_ml.py transcript.txt")
        sys.exit(1)

    transcript_file = sys.argv[1]
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    segments = segment_transcript(
        transcript,
        merge_min_words=6,      # merge very short utterances
        smooth_window=3,        # low-pass filter window for sims
        sim_threshold=0.28,     # boundary threshold (tuneable)
        min_segment_words=35,   # ensure segments have reasonable size
        embed_model="all-MiniLM-L6-v2"
    )

    # Print and save to cases.json for next step
    out = {"cases": segments}
    print("\n=== ML CASE SEGMENTS ===\n")
    for i, s in enumerate(segments, start=1):
        print(f"--- CASE {i} ---\n{s}\n")

    with open("cases.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(segments)} cases to cases.json")
