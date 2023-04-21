import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy
from langchain.vectorstores import Annoy



os.environ["OPENAI_API_KEY"] = "..."

loader = TextLoader("test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

docs[:5]


embeddings_func = HuggingFaceEmbeddings()

vector_store_from_docs = Annoy.from_documents(docs, embeddings_func)


docs = vector_store_from_docs.similarity_search("citizens", k=3)

print(docs[0].page_content[:100])
