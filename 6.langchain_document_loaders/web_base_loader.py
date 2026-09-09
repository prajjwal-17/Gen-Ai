from langchain_community.document_loaders import WebBaseLoader

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'

loader = WebBaseLoader(url)

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content[:1000])
print(docs[0].metadata)