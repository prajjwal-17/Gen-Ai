# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv
# import os

# load_dotenv()

# client = InferenceClient(
#     api_key=os.getenv("HF_TOKEN")
# )

# text = "Delhi is capital of India"

# vector = client.feature_extraction(
#     text,
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )

# print(vector)

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

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