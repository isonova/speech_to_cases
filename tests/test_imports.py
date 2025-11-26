def test_import_main_scripts():
    """Basic smoke test to ensure main scripts import without errors."""
    __import__("transcribe_call")
    __import__("segment_cases_ml")
    __import__("summarize_cases")
    __import__("pipeline")
