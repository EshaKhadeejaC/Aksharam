"""
FULL AUTOMATED PIPELINE

Malayalam scanned PDF → OCR → mal.txt
English digital PDF → extraction → eng.txt
mBART-50 → semantic translation → eng_translated.txt
LaBSE → paragraph alignment → aligned_paragraphs.csv

Uses your original extraction methods with automation and semantic translation
"""

import os
import pytesseract
from pdf2image import convert_from_path
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from translate import MBARTTranslator

# try to load local configuration (ignored by git)
try:
    import config_local as cfg
except ImportError:
    cfg = None


# ==============================
# CONFIGURATION
# ==============================

# default values; may be overridden by config_local
TESSERACT_PATH = getattr(cfg, "TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH   = getattr(cfg, "POPPLER_PATH", r"C:\Users\Acer\Downloads\Release-25.12.0-0\poppler\Library\bin")

MAL_PDF = getattr(cfg, "MAL_PDF", "malayalam_novel.pdf")
ENG_PDF = getattr(cfg, "ENG_PDF", "novel.pdf")

# Use workspace-relative text files (existing under 3_paras/)
MAL_TXT = getattr(cfg, "MAL_TXT", os.path.join("3_paras", "3_para_mal.txt"))
ENG_TXT = getattr(cfg, "ENG_TXT", os.path.join("3_paras", "3_para_eng.txt"))
ENG_TRANSLATED_TXT = getattr(cfg, "ENG_TRANSLATED_TXT", "eng_translated.txt")

OUTPUT_CSV = "aligned_paragraphs.csv"

MODEL_NAME = "sentence-transformers/LaBSE"
WINDOW_SIZE = 5


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==============================
# MALAYALAM OCR EXTRACTION
# ==============================

def extract_malayalam():

    print("\n=== Malayalam OCR Extraction ===")

    images = convert_from_path(
        MAL_PDF,
        dpi=400,
        poppler_path=POPPLER_PATH
    )

    all_text = []

    for i, image in enumerate(images, 1):

        print(f"OCR page {i}/{len(images)}")

        text = pytesseract.image_to_string(
            image,
            lang="mal",
            config="--psm 3"
        )

        all_text.append(text)

    combined = "\n\n".join(all_text)

    with open(MAL_TXT, "w", encoding="utf-8") as f:
        f.write(combined)

    print("Malayalam saved to:", MAL_TXT)


# ==============================
# ENGLISH EXTRACTION
# ==============================

def extract_english():

    print("\n=== English PDF Extraction ===")

    doc = fitz.open(ENG_PDF)

    with open(ENG_TXT, "w", encoding="utf-8") as f:

        for page in doc:

            text = page.get_text("text")

            f.write(text + "\n")

    doc.close()

    print("English saved to:", ENG_TXT)


# ==============================
# PARAGRAPH SPLIT
# ==============================

def read_paragraphs(path):

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    paragraphs = [

        p.strip()
        for p in text.split("\n\n")
        if len(p.strip()) > 10

    ]

    return paragraphs


# ==============================
# ALIGNMENT
# ==============================

def align_paragraphs():

    print("\n=== LaBSE Alignment ===")

    eng_paragraphs = read_paragraphs(ENG_TXT)
    mal_paragraphs = read_paragraphs(MAL_TXT)

    print("English paragraphs:", len(eng_paragraphs))
    print("Malayalam paragraphs:", len(mal_paragraphs))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Generating embeddings...")

    eng_emb = model.encode(
        eng_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    mal_emb = model.encode(
        mal_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )


    sim = cos_sim(eng_emb, mal_emb).cpu().numpy()


    print("Running sequential alignment...")

    aligned = []

    mal_pointer = 0

    for i in range(len(eng_paragraphs)):

        start = mal_pointer
        end = min(len(mal_paragraphs), mal_pointer + WINDOW_SIZE)

        best_score = -1
        best_j = None

        for j in range(start, end):

            score = sim[i][j]

            if score > best_score:

                best_score = score
                best_j = j

        if best_j is not None:

            aligned.append({

                "English": eng_paragraphs[i],
                "Malayalam": mal_paragraphs[best_j],
                "Similarity": float(best_score)

            })

            mal_pointer = best_j


    df = pd.DataFrame(aligned)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Alignment saved to:", OUTPUT_CSV)


# ==============================
# SEMANTIC TRANSLATION
# ==============================

def translate_english_to_malayalam():

    print("\n=== mBART-50 Semantic Translation ===")

    if not os.path.exists(ENG_TXT):
        print(f"English text file {ENG_TXT} not found. Skipping translation.")
        return

    translator = MBARTTranslator()

    translator.translate_file(
        input_path=ENG_TXT,
        output_path=ENG_TRANSLATED_TXT,
        src_lang="en_XX",
        tgt_lang="ml_IN"
    )

    print("Semantic translation completed.")


# ==============================
# MAIN
# ==============================

def main():

    # If text files already exist, skip extraction. Otherwise attempt extraction
    mal_exists = os.path.exists(MAL_TXT)
    eng_exists = os.path.exists(ENG_TXT)

    if mal_exists:
        print(f"Found {MAL_TXT}, skipping Malayalam extraction.")
    else:
        if os.path.exists(MAL_PDF):
            extract_malayalam()
        else:
            print(f"Missing both Malayalam text ({MAL_TXT}) and PDF ({MAL_PDF}). Skipping Malayalam extraction.")

    if eng_exists:
        print(f"Found {ENG_TXT}, skipping English extraction.")
    else:
        if os.path.exists(ENG_PDF):
            extract_english()
        else:
            print(f"Missing both English text ({ENG_TXT}) and PDF ({ENG_PDF}). Skipping English extraction.")

    # Only run alignment if both text files are present
    if os.path.exists(MAL_TXT) and os.path.exists(ENG_TXT):
        align_paragraphs()
        translate_english_to_malayalam()
    else:
        print("Alignment and translation skipped because input text files are missing.")


if __name__ == "__main__":

    main()
