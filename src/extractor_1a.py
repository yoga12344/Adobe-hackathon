import fitz
import os
import json
import sys
import fasttext
from typing import Dict, Any, List, Optional
from src.utils import detect_language, analyze_font_sizes, clean_text
from src.components import Line, Scorer

class PDFOutlineExtractor:
    CONFIG = {
        'scoring': {
            'base_score_line_length': 5, 'font_size_p99': 20, 'font_size_p95': 15,
            'font_size_p90': 10, 'font_weight_bold': 5, 'numbered_list_bonus': 10,
            'ends_with_period_penalty': -10,
        },
        'thresholds': {'h1': 25, 'h2': 20, 'h3': 15}
    }

    def __init__(self, pdf_path: str, model_path: str = "models/lid.176.bin"):
        self.doc: Optional[fitz.Document] = None
        try:
            self.doc = fitz.open(pdf_path)
            self.lang_model = fasttext.load_model(model_path)
        except Exception as e:
            print(f"Error opening file {pdf_path}: {e}", file=sys.stderr)

    def extract_structure(self) -> Dict[str, Any]:
        if not self.doc:
            return {"title": "Invalid Document", "language": "unknown", "outline": []}

        language = detect_language(self.doc, self.lang_model)
        font_stats = analyze_font_sizes(self.doc)
        scorer = Scorer(font_stats, language, self.CONFIG)
        
        title = os.path.basename(self.doc.name).replace('.pdf', '')
        max_title_score = 0
        outline, seen_lines = [], set()

        for page_num, page in enumerate(self.doc, start=1):
            table_bboxes = get_table_bboxes(page)
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get('type') != 0: continue
                for line_info in block.get("lines", []):
                    line = Line(line_info, page_num)
                    
                    if not line.is_valid() or line.text in seen_lines or line.is_in_bboxes(table_bboxes):
                        continue

                    score = scorer.score(line)
                    
                    # Title detection logic
                    if page_num == 1 and score > max_title_score:
                        title = clean_text(line.text)
                        max_title_score = score

                    level = scorer.get_level(score)
                    if level:
                        outline.append({"level": level, "text": clean_text(line.text), "page": page_num})
                        seen_lines.add(line.text)
                        
        return {"title": title, "language": language, "outline": outline}

def main(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    for filename in pdf_files:
        pdf_path = os.path.join(input_dir, filename)
        print(f"Processing {filename} for Round 1A...")
        
        try:
            extractor = PDFOutlineExtractor(pdf_path)
            if not extractor.doc:
                print(f"Skipping {filename} due to initialization error.", file=sys.stderr)
                continue
            
            result = extractor.extract_structure()
            
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully created outline for {filename}")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python extractor_1a.py <input_folder> <output_folder>", file=sys.stderr)
