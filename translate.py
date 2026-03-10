"""
mBART-50 Translation Pipeline for Semantic Meaning Preservation

Translates text while maintaining semantic meaning using Facebook's mBART-50 model.
Supports 50 languages with high-quality translation.
"""

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
import torch

MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Language codes for mBART-50
LANG_CODES = {
    "english": "en_XX",
    "malayalam": "ml_IN",
    "hindi": "hi_IN",
    "tamil": "ta_IN",
    "telugu": "te_IN",
    "kannada": "kn_IN",
    "bengali": "bn_IN",
    "gujarati": "gu_IN",
    "punjabi": "pa_IN",
    "marathi": "mr_IN",
    "oriya": "or_IN",
    "assamese": "as_IN",
    "urdu": "ur_PK",
    "sanskrit": "sa_IN",
    "nepali": "ne_NP",
    "sinhala": "si_LK",
    "arabic": "ar_AR",
    "french": "fr_XX",
    "german": "de_DE",
    "spanish": "es_XX",
    "italian": "it_IT",
    "portuguese": "pt_XX",
    "russian": "ru_RU",
    "chinese": "zh_CN",
    "japanese": "ja_XX",
    "korean": "ko_KR"
}

class MBARTTranslator:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")

        # Load model and tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(MODEL)
        self.model = MBartForConditionalGeneration.from_pretrained(MODEL)

        if self.device == 0:
            self.model = self.model.to("cuda")

        # Create translation pipeline
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def translate_text(self, text, src_lang="en_XX", tgt_lang="ml_IN", max_length=1024):
        """
        Translate text from source language to target language.

        Args:
            text (str): Text to translate
            src_lang (str): Source language code (e.g., 'en_XX')
            tgt_lang (str): Target language code (e.g., 'ml_IN')
            max_length (int): Maximum length of translated text

        Returns:
            str: Translated text
        """
        # Set language tokens
        self.tokenizer.src_lang = src_lang

        # Translate
        result = self.translator(text, max_length=max_length, tgt_lang=tgt_lang)

        return result[0]["translation_text"]

    def translate_file(self, input_path, output_path, src_lang="en_XX", tgt_lang="ml_IN"):
        """
        Translate a text file paragraph by paragraph to preserve semantic meaning.

        Args:
            input_path (str): Path to input text file
            output_path (str): Path to output translated file
            src_lang (str): Source language code
            tgt_lang (str): Target language code
        """
        print(f"Translating {input_path} from {src_lang} to {tgt_lang}...")

        # Read input text
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into paragraphs to maintain semantic chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        translated_paragraphs = []

        for i, para in enumerate(paragraphs, 1):
            print(f"Translating paragraph {i}/{len(paragraphs)}...")
            translated = self.translate_text(para, src_lang, tgt_lang)
            translated_paragraphs.append(translated)

        # Write translated text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_paragraphs))

        print(f"Translation complete. Saved to: {output_path}")

    def get_lang_code(self, language_name):
        """Get mBART language code from language name."""
        return LANG_CODES.get(language_name.lower(), "en_XX")


# Convenience function for easy use
def translate_file(input_path, output_path, src_lang="en_XX", tgt_lang="ml_IN"):
    """Convenience function to translate a file."""
    translator = MBARTTranslator()
    translator.translate_file(input_path, output_path, src_lang, tgt_lang)


if __name__ == "__main__":
    # Example usage
    translator = MBARTTranslator()

    # Translate English to Malayalam
    translator.translate_file(
        input_path="english_full.txt",
        output_path="english_to_malayalam.txt",
        src_lang="en_XX",
        tgt_lang="ml_IN"
    )