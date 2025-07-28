import re
from typing import Dict, Any, Optional

class Line:
    """Represents a single line of text from a PDF with its properties."""
    def __init__(self, line_info: Dict[str, Any], page_num: int):
        self.page_num = page_num
        self.spans = line_info.get("spans", [])
        self.text = "".join(s["text"] for s in self.spans).strip()
        self.bbox = line_info.get("bbox", None)
        self.font_size = self.spans[0]["size"] if self.spans else 0
        self.font_name = self.spans[0]["font"] if self.spans else ""
        self.is_bold = "bold" in self.font_name.lower()

    def is_valid(self) -> bool:
        """Checks if the line is valid for processing."""
        return bool(self.text and self.spans and self.bbox)

    def is_in_bboxes(self, bboxes: list) -> bool:
        """Checks if the line is inside any of the given bounding boxes."""
        if not self.bbox:
            return False
        line_rect = fitz.Rect(self.bbox)
        for bbox in bboxes:
            if bbox.contains(line_rect):
                return True
        return False

class Scorer:
    """Calculates a score for a line to determine if it's a heading."""
    def __init__(self, font_stats: Dict[str, float], language: str, config: Dict[str, Any]):
        self.font_stats = font_stats
        self.language = language
        self.config = config

    def score(self, line: Line) -> float:
        """Calculates the score for a given line."""
        score = 0.0
        cfg = self.config['scoring']
        
        # Score based on line length
        score += max(0, 1 - len(line.text) / 100.0) * cfg['base_score_line_length']

        # Score based on font size
        if line.font_size >= self.font_stats.get('p99', 99):
            score += cfg['font_size_p99']
        elif line.font_size >= self.font_stats.get('p95', 30):
            score += cfg['font_size_p95']
        elif line.font_size >= self.font_stats.get('p90', 20):
            score += cfg['font_size_p90']

        # Score for bold font
        if line.is_bold:
            score += cfg['font_weight_bold']

        # Bonus for numbered lists
        if (self.language == 'ja' and re.match(r"^[第]?[一二三四五六七八九十百千\d]+", line.text)) or \
           re.match(r"^\d+(\.\d+)*", line.text):
            score += cfg['numbered_list_bonus']
            
        # Penalty for ending with a period
        if line.text.endswith('.'):
            score += cfg['ends_with_period_penalty']
            
        return score

    def get_level(self, score: float) -> Optional[str]:
        """Determines the heading level from a score."""
        cfg = self.config['thresholds']
        if score >= cfg['h1']:
            return "H1"
        if score >= cfg['h2']:
            return "H2"
        if score >= cfg['h3']:
            return "H3"
        return None
