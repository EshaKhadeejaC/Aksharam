import os
import json
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel

def generate_translations(model, tokenizer, texts, device, max_length=128):
    # Set target language
    tokenizer.src_lang = "en_XX"
    forced_target_id = tokenizer.lang_code_to_id["ml_IN"]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=forced_target_id,
            max_length=max_length,
            num_beams=4 # Basic beam search for better translations
        )
        
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def main():
    base_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    adapter_path = "./aksharam_model_final"

    if not os.path.exists(adapter_path):
        print(f"Error: {adapter_path} not found. Please run 02_train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer and base model...")
    tokenizer = MBart50TokenizerFast.from_pretrained(base_model_name, src_lang="en_XX")
    base_model = MBartForConditionalGeneration.from_pretrained(base_model_name)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    print("\n" + "="*50)
    print("Aksharam Translation System Loaded Successfully!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nEnter English sentence: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            if not user_input.strip():
                continue
                
            print("Translating...")
            hypothesis = generate_translations(model, tokenizer, [user_input], device)[0]
            
            print(f"Malayalam Translation: {hypothesis}")
            with open("translations.txt", "a", encoding="utf-8") as f:
               f.write(user_input + " -> " + hypothesis + "\n")
        except KeyboardInterrupt:
            break
            
    print("\nExited Aksharam.")

if __name__ == "__main__":
    main()
