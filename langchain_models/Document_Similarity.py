# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from torch import embedding

# load_dotenv()

# embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

# documents =  [
#     " Cristiano Ronaldo is one of the most prolific goal scorers in football history. "
#     " Real Madrid is one of the most successful clubs in UEFA Champions League history. "
#     " Lionel Messi is known for his dribbling, playmaking, and goal-scoring ability. "
#     " The FIFA World Cup is the biggest international tournament in football. "
#     " Midfielders play a crucial role in controlling possession and creating chances for their team. "
# ]

# query = "Tell me about Cristiano Ronaldo"

# doc_embeddings = embedding.embed_documents(documents)
# query_embeddings = embedding.embed_query(query)

# scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

# print(query)
# print(documents[index])
# print("Simialrity Score is  : " , score )

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

documents = [

    "Cristiano Ronaldo is one of the most prolific goal scorers in football history.",

    "Real Madrid is one of the most successful clubs in UEFA Champions League history.",

    "Lionel Messi is known for his dribbling, playmaking, and goal-scoring ability.",

    "The FIFA World Cup is the biggest international tournament in football.",

    "Midfielders play a crucial role in controlling possession and creating chances for their team."

]

query = "Tell me about Cristiano Ronaldo"

doc_embeddings = client.feature_extraction(
    documents,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

query_embeddings = client.feature_extraction(
    query,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)

print(documents[index])

print("Similarity Score is : ", score)