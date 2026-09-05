from huggingface_hub import InferenceClient
from dotenv  import load_dotenv

load_dotenv()

client = InferenceClient()

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": """
            Analyze this review and return a summary and sentiment.

            The hardware is great, but the software feels bloated.
            There are too many pre-installed apps that I can't remove.
            The UI looks outdated compared to other brands.
            """
        }
    ],
    max_tokens=100,
)

print(response.choices[0].message)