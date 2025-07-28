#!/bin/bash

# --- Round 1A Execution ---
# Processes all PDFs in /app/input/round_1a and saves JSON outlines to /app/output/round_1a
echo "--- Running Round 1A: PDF Outline Extraction ---"
python3 src/extractor_1a.py /app/input/round_1a /app/output/round_1a


# --- Round 1B Execution (Uncomment to run) ---
# Analyzes documents in /app/input/round_1b based on the persona and job description
# and saves the relevance analysis to /app/output/round_1b
#
# echo "--- Running Round 1B: Persona-Driven Analysis ---"
# python3 src/analyzer_1b.py \
#   /app/input/round_1b/ \
#   /app/input/round_1b/persona.txt \
#   /app/input/round_1b/job_to_be_done.txt \
#   /app/output/round_1b/
