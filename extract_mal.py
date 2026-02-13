"""
Malayalam OCR using Tesseract
Extracts Malayalam text from PDF and saves to text file
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os


class MalayalamOCR:
    def __init__(self, tesseract_path=None):
       
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Check if Malayalam language data is available
        self.verify_malayalam_support()
    
    
    def pdf_to_images(self, pdf_path, dpi=300):
        
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"✓ Converted {len(images)} pages to images")
        return images
    
    def image_to_text(self, image, lang='mal', config=''):
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text
    
    def process_pdf(self, pdf_path, output_file='malayalam_novel.txt', 
                   lang='mal', dpi=300, config=''):
        
        print(f"\n{'='*60}")
        print(f"Starting OCR for: {pdf_path}")
        print(f"{'='*60}\n")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi)
        
        # Extract text from each page
        all_text = []
        for i, image in enumerate(images, 1):
            print(f"Processing page {i}/{len(images)}...")
            text = self.image_to_text(image, lang=lang, config=config)
            all_text.append(text)
            print(f"✓ Extracted {len(text)} characters from page {i}")
        
        # Combine all text
        combined_text = '\n\n'.join(all_text)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"\n{'='*60}")
        print(f"✅ OCR Complete!")
        print(f"{'='*60}")
        print(f"Total pages processed: {len(images)}")
        print(f"Total characters extracted: {len(combined_text)}")
        print(f"Output saved to: {output_file}")
        
        return combined_text
    
    def process_image_file(self, image_path, output_file='malayalam_text.txt',
                          lang='mal', config=''):
        
        print(f"Processing image: {image_path}")
        
        # Open image
        image = Image.open(image_path)
        
        # Extract text
        text = self.image_to_text(image, lang=lang, config=config)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✅ Extracted {len(text)} characters")
        print(f"Output saved to: {output_file}")
        
        return text
    
    def process_images_folder(self, folder_path, output_file='malayalam_novel.txt',
                             lang='mal', config=''):
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_files = []
        
        for file in sorted(os.listdir(folder_path)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        all_text = []
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            image = Image.open(image_path)
            text = self.image_to_text(image, lang=lang, config=config)
            all_text.append(text)
            print(f"✓ Extracted {len(text)} characters")
        
        # Combine and save
        combined_text = '\n\n'.join(all_text)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"\n✅ Processed {len(image_files)} images")
        print(f"Total characters: {len(combined_text)}")
        print(f"Output saved to: {output_file}")
        
        return combined_text


def main():
    """Main execution function"""
    
    # Configuration
    PDF_PATH = 'malayalam_novel.pdf'  # Path to your Malayalam PDF
    OUTPUT_FILE = 'malayalam_novel.txt'  # Output text file
    DPI = 300  # Resolution (300 is good balance, 400-600 for better quality)
    
    TESSERACT_PATH = None  # Set to None for Linux/Mac
    
    # Initialize OCR
    ocr = MalayalamOCR(tesseract_path=TESSERACT_PATH)
    
    # Process PDF
    text = ocr.process_pdf(
        pdf_path=PDF_PATH,
        output_file=OUTPUT_FILE,
        lang='mal',  # Malayalam language code
        dpi=DPI
    )
    
    print("\n" + "="*60)
    print("Preview of extracted text (first 500 characters):")
    print("="*60)
    print(text[:500])
    print("...")


# Alternative usage examples
def example_image_processing():
    """Example: Process a single image file"""
    ocr = MalayalamOCR()
    text = ocr.process_image_file(
        image_path='page1.jpg',
        output_file='page1_text.txt',
        lang='mal'
    )


def example_folder_processing():
    """Example: Process all images in a folder"""
    ocr = MalayalamOCR()
    text = ocr.process_images_folder(
        folder_path='malayalam_pages/',
        output_file='malayalam_novel.txt',
        lang='mal'
    )


if __name__ == "__main__":
    main()
