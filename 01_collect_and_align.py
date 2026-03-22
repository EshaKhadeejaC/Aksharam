import json
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading LaBSE model...")
    model = SentenceTransformer(
        "sentence-transformers/LaBSE",
        device=device
    )

    en_sentences = []
    ml_sentences = []

    print("Collecting datasets...")

    # -------------------------------------------------
    # 1. OPUS-100
    # -------------------------------------------------
    try:
        opus = load_dataset("opus100", "en-ml", split="train[:40000]")

        before = len(en_sentences)

        for item in tqdm(opus, desc="OPUS-100"):

            t = item.get("translation", {})

            en = t.get("en")
            ml = t.get("ml")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"OPUS added: {after - before}")

    except Exception as e:
        print("OPUS load failed:", e)


    # -------------------------------------------------
    # 2. Hemanth dataset
    # -------------------------------------------------
    try:
        hemanth = load_dataset(
            "Hemanth-thunder/english-to-malayalam-mt",
            split="train[:50000]"
        )

        before = len(en_sentences)

        for item in tqdm(hemanth, desc="Hemanth dataset"):

            en = item.get("english") or item.get("en")
            ml = item.get("malayalam") or item.get("ml")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"Hemanth added: {after - before}")

    except Exception as e:
        print("Hemanth dataset failed:", e)


    # -------------------------------------------------
    # 3. Samanantar dataset
    # -------------------------------------------------
    try:
        samanantar = load_dataset(
            "ai4bharat/samanantar",
            "ml",
            split="train[:100000]"
        )

        before = len(en_sentences)

        for item in tqdm(samanantar, desc="Samanantar dataset"):

            en = item.get("src")
            ml = item.get("tgt")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"Samanantar added: {after - before}")

    except Exception as e:
        print("Samanantar dataset failed:", e)


    print("\nTotal collected pairs:", len(en_sentences))


    # -------------------------------------------------
    # LaBSE semantic filtering
    # -------------------------------------------------

    threshold = 0.60
    batch_size = 64
    aligned_pairs = []

    print("Filtering pairs using semantic similarity...")

    for i in tqdm(range(0, len(en_sentences), batch_size)):

        batch_en = en_sentences[i:i + batch_size]
        batch_ml = ml_sentences[i:i + batch_size]

        en_emb = model.encode(
            batch_en,
            convert_to_tensor=True,
            batch_size=32
        )

        ml_emb = model.encode(
            batch_ml,
            convert_to_tensor=True,
            batch_size=32
        )

        scores = util.cos_sim(en_emb, ml_emb).diag()

        for j, score in enumerate(scores):

            if score.item() >= threshold:

                aligned_pairs.append({
                    "en": batch_en[j],
                    "ml": batch_ml[j],
                    "score": float(score)
                })


    print(
        f"Kept {len(aligned_pairs)} high-quality pairs "
        f"out of {len(en_sentences)}"
    )


    # -------------------------------------------------
    # Save dataset
    # -------------------------------------------------

    with open("aligned_dataset.json", "w", encoding="utf-8") as f:
        json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)

    print("Dataset saved to aligned_dataset.json")


if __name__ == "__main__":
    main()