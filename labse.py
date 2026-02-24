"""
Paragraph Alignment using LaBSE + Hungarian Algorithm

Reads:
    data/eng.txt
    data/mal.txt

Outputs:
    output/aligned_paragraphs.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch


# ==============================
# CONFIGURATION
# ==============================

MODEL_NAME = "sentence-transformers/LaBSE"
SIMILARITY_THRESHOLD = 0.75

ENG_FILE_PATH = "eng.txt"
MAL_FILE_PATH = "mal.txt"
OUTPUT_CSV_PATH = "output/aligned_paragraphs.csv"


# ==============================
# UTILITY FUNCTIONS
# ==============================

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


# ==============================
# MAIN PIPELINE
# ==============================

def main():

    print("Step 1: Reading files...")
    eng_text = read_text_file(ENG_FILE_PATH)
    mal_text = read_text_file(MAL_FILE_PATH)

    eng_paragraphs = split_into_paragraphs(eng_text)
    mal_paragraphs = split_into_paragraphs(mal_text)

    print("English paragraphs:", len(eng_paragraphs))
    print("Malayalam paragraphs:", len(mal_paragraphs))

    if len(eng_paragraphs) == 0 or len(mal_paragraphs) == 0:
        raise ValueError("One of the files has no paragraphs.")

    print("Step 2: Loading LaBSE model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Using device:", device)

    print("Step 3: Generating embeddings...")
    eng_embeddings = model.encode(
        eng_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    mal_embeddings = model.encode(
        mal_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print("Step 4: Computing similarity matrix...")
    sim_matrix = cos_sim(eng_embeddings, mal_embeddings)
    sim_np = sim_matrix.cpu().numpy()

    print("Similarity matrix shape:", sim_np.shape)

    print("Step 5: Running Hungarian Algorithm...")

    cost_matrix = -sim_np
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    print("Total assignments:", len(row_ind))

    print("Step 6: Applying similarity threshold...")

    aligned_pairs = []

    for r, c in zip(row_ind, col_ind):
        score = sim_np[r][c]

        if score >= SIMILARITY_THRESHOLD:
            aligned_pairs.append({
                "English": "<en_xx> " + eng_paragraphs[r],
                "Malayalam": "<ml_xx> " + mal_paragraphs[c],
                "Similarity": float(score)
            })

    print("Final aligned pairs after threshold:", len(aligned_pairs))

    print("Step 7: Writing results to CSV...")

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    df = pd.DataFrame(aligned_pairs)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Alignment complete.")
    print("CSV saved at:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
