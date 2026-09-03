from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

client = InferenceClient()
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation")
    
model = ChatHuggingFace(llm=llm)

#schema 
class Review(TypedDict):
    summary : str
    sentiment : str
    
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""
                      The hardware is great, but the software feels bloated . There are too many pre-installled apps that i cant remove.
                      Also the ui looks outdated compared to other brands . Hoping for a software update to fix this
                      """)

print(result)