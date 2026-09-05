from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient()

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = client.feature_extraction(
    documents,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print(str(vector))