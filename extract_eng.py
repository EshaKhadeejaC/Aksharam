import fitz  

pdf_path = "novel.pdf"        
output_txt = "output_preview.txt"  

doc = fitz.open(pdf_path)
lines = []

for page in doc:
    text = page.get_text("text")
    page_lines = text.splitlines()
    lines.extend(page_lines)
    if len(lines) >= 500:  
        break

doc.close()

with open(output_txt, "w", encoding="utf-8") as f:
    for line in lines[:500]:
        f.write(line + "\n")

print(f"Extracted {len(lines[:500])} lines and saved to '{output_txt}'")
