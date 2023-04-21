import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Annoy
import pickle

os.environ["OPENAI_API_KEY"] = "..."

game_captions = []
with open("Image2Paragraph/image2textMindcraft/141_212_108_99_20210325_121618.pkl", "rb") as game_file:
    game_captions = pickle.load(game_file)
print(len(game_captions))
'''
TODO: Check if this is needed, maybe not given all our data is in lists?

loader = TextLoader("test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

docs[:5]
'''


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}

openai = OpenAIEmbeddings(openai_api_key="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")
frame_index = Annoy.from_texts(game_captions, openai)
frame_index.save_local("game_stores/141_212_108_99_20210325_121618")
print(frame_index)
top_k_frames = frame_index.similarity_search("Has your partner made Blue Wool?", k=3)

print(top_k_frames)
print("finish")
