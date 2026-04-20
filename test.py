import os
# 🚨 THIS MUST BE AT THE VERY TOP, BEFORE ANY OTHER IMPORTS! 🚨
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Now we can safely import the Hugging Face libraries
from langchain_huggingface import HuggingFaceEmbeddings
import config
import time

def download_and_test_model():
    print(f"🚀 Starting download for embedding model: {config.EMBEDDING_MODEL_NAME}")
    print("🌍 Using HF Mirror Endpoint to bypass connection issues...")
    
    start_time = time.time()
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"\n❌ Error downloading the model: {e}")
        return

    print("\n✅ Model downloaded to local cache!")
    print("🔄 Generating a test embedding to verify...")
    
    test_text = "This is a test sentence to verify the embeddings are working locally."
    vector = embeddings.embed_query(test_text)
    
    end_time = time.time()
    
    print(f"✅ Success! Test embedding generated with dimension: {len(vector)}")
    print(f"⏱️ Total time taken: {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    download_and_test_model()