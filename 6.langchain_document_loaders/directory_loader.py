from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()   # normal loader
print(len(docs)) # works with normal loader


docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
    
    
