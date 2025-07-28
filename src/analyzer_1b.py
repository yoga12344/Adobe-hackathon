import fitz
import os
import json
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from extractor_1a import PDFOutlineExtractor
from datetime import datetime
from typing import Dict, Any, List

class PersonaDocumentAnalyzer:
    def __init__(self, model_path: str = "models/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)

    def analyze(self, doc_paths: List[str], persona_text: str, job_text: str) -> Dict[str, Any]:
        # Combine persona and job for a single query
        query = f"Persona: {persona_text}. Task: {job_text}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        all_sections = []
        for doc_path in doc_paths:
            print(f"Analyzing {os.path.basename(doc_path)} for Round 1B...")
            # 1. Extract outline using Round 1A logic
            extractor = PDFOutlineExtractor(doc_path)
            structure = extractor.extract_structure()
            
            # 2. Extract text content for each section
            doc = fitz.open(doc_path)
            for i, heading in enumerate(structure['outline']):
                start_page = heading['page'] - 1
                end_page = structure['outline'][i+1]['page'] - 1 if i + 1 < len(structure['outline']) else len(doc)
                
                section_text = heading['text']
                for page_num in range(start_page, end_page + 1):
                    # A simple text extraction for content under a heading
                    section_text += doc[page_num].get_text()

                all_sections.append({
                    'document': os.path.basename(doc_path),
                    'page': heading['page'],
                    'section_title': heading['text'],
                    'content': section_text
                })

        # 3. Vectorize all sections and calculate relevance
        if not all_sections: return {}
        
        section_embeddings = self.model.encode([s['content'] for s in all_sections], convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, section_embeddings)[0]

        for i, section in enumerate(all_sections):
            section['relevance_score'] = cosine_scores[i].item()
        
        # 4. Rank sections by relevance
        ranked_sections = sorted(all_sections, key=lambda x: x['relevance_score'], reverse=True)
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
        
        return self._format_output(doc_paths, persona_text, job_text, ranked_sections)

    def _format_output(self, docs: List[str], persona: str, job: str, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "metadata": {
                "input_documents": [os.path.basename(d) for d in docs],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": s['document'],
                    "page_number": s['page'],
                    "section_title": s['section_title'],
                    "importance_rank": s['importance_rank'],
                    # Sub-section analysis can be implemented here
                    "refined_text": s['content'][:300] + "..." # Truncated for brevity
                } for s in sections
            ]
        }

def main(doc_folder: str, persona_file: str, job_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    doc_paths = [os.path.join(doc_folder, f) for f in os.listdir(doc_folder) if f.lower().endswith('.pdf')]
    with open(persona_file, 'r', encoding='utf-8') as f:
        persona_text = f.read()
    with open(job_file, 'r', encoding='utf-8') as f:
        job_text = f.read()
        
    analyzer = PersonaDocumentAnalyzer()
    result = analyzer.analyze(doc_paths, persona_text, job_text)
    
    out_path = os.path.join(output_dir, "analysis_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Analysis complete. Output saved to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
