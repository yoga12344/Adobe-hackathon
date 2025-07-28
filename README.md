# Adobe Hackathon: Connecting the Dots

This project is a solution for the Adobe "Connecting the Dots" Hackathon, providing implementations for both Round 1A and Round 1B.

## Solution Overview

The project is structured into two main components:

1.  **Round 1A: PDF Outline Extractor**: A Python script that extracts a structured outline (Title, H1, H2, H3) from PDF documents. It uses a feature-based scoring model and `fastText` for language detection to support multilingual documents.
2.  **Round 1B: Persona-Driven Document Analyzer**: A system that reuses the 1A extractor and then applies a `sentence-transformer` model to rank document sections based on their semantic relevance to a given user persona and job description.

### Round 1A Approach

- **Modular Design**: The solution is broken down into logical components to improve readability and maintainability:
    - `extractor_1a.py`: The main script that orchestrates the extraction process.
    - `utils.py`: Contains helper functions for language detection, font analysis, and text cleaning.
    - `components.py`: Defines the `Line` and `Scorer` classes, which encapsulate the core logic for processing and scoring text lines.
- **Language Detection**: The script first identifies the document's language (`en`, `ja`, etc.) using the compact `fasttext/lid.176.bin` model.
- **Feature-Based Scoring**: Instead of relying only on font sizes, each line is scored based on a combination of features:
    - Font size (relative to percentiles)
    - Font weight (bold)
    - Line length
    - Presence of numbering (standard and Japanese)
- **Classification**: Lines with scores exceeding configured thresholds are classified as H1, H2, or H3.
- **Output**: The result is a clean JSON file containing the document's title, detected language, and a hierarchical outline.

### Round 1B Approach

- **Structure Extraction**: The system first processes all input documents using the Round 1A extractor to get structured outlines.
- **Semantic Embedding**: It uses the `all-MiniLM-L6-v2` sentence-transformer model to generate vector embeddings for:
    1. The combined `persona` and `job-to-be-done` text.
    2. The text content of each section extracted in the previous step.
- **Relevance Ranking**: It calculates the cosine similarity between the persona/job vector and each section vector. This score determines the section's relevance.
- [cite_start]**Output**: The final output is a single JSON file containing metadata and a list of all sections from all documents, ranked by their importance to the user's task[cite: 149].

---

## Build and Run Instructions

**Prerequisites**: Docker must be installed.

**1. Setup**:
- Create a `models` directory in the project root.
- Download the `lid.176.bin` model and place it in `models/`.
- Download and unzip the `all-MiniLM-L6-v2` model and place the folder in `models/`.

**2. Build the Docker Image**:
```sh
docker build --platform linux/amd64 -t adobe-hackathon-solution .
