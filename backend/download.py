from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
SAVE_PATH = "./models/clip"

print("⬇️ Downloading model...")
model = CLIPModel.from_pretrained(MODEL_NAME)

print("⬇️ Downloading processor...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("💾 Saving locally...")
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("✅ Done. Model saved to:", SAVE_PATH)