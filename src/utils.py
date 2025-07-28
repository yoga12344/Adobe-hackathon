import fitz
import numpy as np
import fasttext
import re
from typing import Dict, Any, List, Optional, Tuple

def clean_text(text: str) -> str:
    """Removes extra whitespace and non-printable characters from text."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return "".join(char for char in text if char.isprintable())

def detect_language(doc: fitz.Document, model: fasttext.FastText._FastText) -> str:
    """Detects the primary language of the document."""
    text_sample = "".join(page.get_text() for page in doc.pages(stop=5))
    cleaned_text = text_sample.replace("\n", " ").strip()
    if not cleaned_text:
        return "en"  # Default to English
    predictions = model.predict(cleaned_text, k=1)
    lang = predictions[0][0].replace("__label__", "")
    print(f"Detected language: {lang}")
    return lang

def analyze_font_sizes(doc: fitz.Document) -> Dict[str, float]:
    """Analyzes font sizes across the document to find key percentiles."""
    sizes = [
        s["size"]
        for p in doc
        for b in p.get_text("dict")["blocks"] if b.get('type') == 0
        for l in b.get("lines", [])
        for s in l.get("spans", [])
    ]
    if not sizes:
        return {'p99': 20, 'p95': 15, 'p90': 10}  # Default values
    
    stats = {
        'p99': np.percentile(sizes, 99),
        'p95': np.percentile(sizes, 95),
        'p90': np.percentile(sizes, 90),
        'median': np.median(sizes)
    }
    return stats

def get_table_bboxes(page: fitz.Page) -> List[fitz.Rect]:
    """Finds and returns the bounding boxes of all tables on a page."""
    table_finder = page.find_tables()
    return [table.bbox for table in table_finder]
