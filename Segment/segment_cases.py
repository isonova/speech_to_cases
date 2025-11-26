import re

# Patterns for detecting case boundaries
BOUNDARY_PATTERNS = [
    r"\bcase\s+\d+",
    r"\bcase\s+one\b",
    r"\bcase\s+two\b",
    r"\bcase\s+three\b",
    r"\bnext\s+case\b",
    r"\bnext\s+patient\b",
    r"\bsecond\s+patient\b",
    r"\bthird\s+patient\b",
]

def is_boundary(sentence: str) -> bool:
    """Return True if sentence contains case boundary keywords."""
    sentence = sentence.lower()
    for pattern in BOUNDARY_PATTERNS:
        if re.search(pattern, sentence):
            return True
    return False

def split_into_sentences(text: str):
    """Naive sentence splitter – later we switch to spaCy."""
    return re.split(r'(?<=[.!?])\s+', text.strip())

def segment_transcript(transcript: str):
    """Split transcript into case segments."""
    sentences = split_into_sentences(transcript)
    cases = []
    current = []

    for s in sentences:
        if is_boundary(s) and current:
            cases.append(" ".join(current))
            current = [s]
        else:
            current.append(s)

    if current:
        cases.append(" ".join(current))

    return cases

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python segment_cases.py <transcript_file.txt>")
        exit(1)

    text_file = sys.argv[1]
    with open(text_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    cases = segment_transcript(transcript)

    print("\n=== CASE SEGMENTS ===\n")
    for i, c in enumerate(cases, start=1):
        print(f"\n--- CASE {i} ---\n")
        print(c)
