"""
mBART-50 Pipeline for Aligned Paragraphs

Reads the LaBSE-aligned CSV, translates English paragraphs to Malayalam using mBART-50,
and generates a new CSV with translations for semantic comparison.
"""

import pandas as pd
from translate import MBARTTranslator
import os

def process_aligned_csv(input_csv="aligned_paragraphs.csv", output_csv="translated_aligned.csv"):
    """
    Process the aligned paragraphs CSV:
    - Read English and Malayalam columns
    - Translate English to Malayalam using mBART-50
    - Add translated column and save new CSV
    """

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run the alignment pipeline first.")
        return

    print(f"Loading aligned data from {input_csv}...")
    df = pd.read_csv(input_csv)

    if "English" not in df.columns:
        print("Error: 'English' column not found in CSV.")
        return

    print(f"Found {len(df)} aligned pairs.")

    # Initialize translator
    translator = MBARTTranslator()

    # Translate each English paragraph
    translated_malayalam = []
    for i, eng_text in enumerate(df["English"], 1):
        print(f"Translating paragraph {i}/{len(df)}...")
        translated = translator.translate_text(eng_text, src_lang="en_XX", tgt_lang="ml_IN")
        translated_malayalam.append(translated)

    # Add new column
    df["Translated_Malayalam"] = translated_malayalam

    # Save to new CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Translated CSV saved to: {output_csv}")

    # Optional: Print some statistics
    print(f"\nProcessed {len(df)} paragraphs.")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    process_aligned_csv()