import os
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import pickle
from glob import glob

hf = HuggingFaceEmbeddings()
k = 2

def chagpt_questioning(agent, top_k_q1, question1):
    return

for game_path in glob("Image2Paragraph/image2textMindcraft/*.pkl"):
    game_captions = [] # Store captions for each frame
    game_agent = None # Agent that will reason over current game
    with open(game_path, "rb") as game_file:
        game_captions = pickle.load(game_file)
    print(game_path, len(game_captions))

    game_index = Chroma.from_texts(["Nothing"],hf)
    game_recency = []
    questions = [None]*len(game_captions) # Array to store questions, no question = None

    for i_s, scene in enumerate(game_captions):
        # Game_index for q1,q2; game_recency for time based retrieval for q3 
        game_index.add_texts([scene])
        game_recency.append(scene)

        # If this timestep has an associated question, ask chatgpt the questions
        if questions[i_s]:
            top_k_q1 = game_index.similarity_search(question1)[:k]
            generated_answer_q1 = chatgpt_questioning(agent, top_k_q1, question1)

            top_k_q2 = game_index.similarity_search(question2)[:k]
            generated_answer_q2 = chatgpt_questioning(agent, top_k_q2, question2)

            generated_answer_q3 = chatgpt_questioning(agent, game_recency[:k], question3)

            print(generated_answer_q1, generated_answer_q2, generated_answer_q3)
            # TODO: Get answer from result

    print(game_path, "finish")
