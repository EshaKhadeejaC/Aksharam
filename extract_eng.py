import fitz

# load local config if available
try:
    import config_local as cfg
except ImportError:
    cfg = None

pdf_path = getattr(cfg, "ENG_PDF", "novel.pdf") if cfg else "novel.pdf"
output_txt = getattr(cfg, "ENG_TXT", "english_full.txt") if cfg else "english_full.txt"

doc = fitz.open(pdf_path)

with open(output_txt, "w", encoding="utf-8") as f:
    for page in doc:
        text = page.get_text("text")
        f.write(text + "\n")

doc.close()

print("English text extracted successfully")
