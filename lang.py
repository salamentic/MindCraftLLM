import os
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import pickle
from glob import glob

# Agent imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

hf = HuggingFaceEmbeddings()
k = 2

def chatgpt_questioning(agent, top_k_history, question):
    sys_msg = f"You are a player in a minecraft experiment. You and your partner are given limited crafting knowledge. You are given scene description \
            from a vision model that cannot use minecraft terms to explain scenes.\n \
            Goal:Generalize scene descriptions into MineCraft terms. Then, answer questions based on understanding and options given. \
            \n"

    hmn_msg = f"\nScenes:\n{top_k_history}\nQuestion: {question}\n Options: [YES, NO, MAYBE] \n Answer:"
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=hmn_msg)]
    #print(agent(messages))
    print(top_k_history)
    return

def main():
    for game_path in glob("Image2Paragraph/image2textMindcraft/*.pkl"):
        game_captions = [] # Store captions for each frame
        # IMPORTANT! REPLACE WITH API KEY!
        # MAKE A FILE CALLED api_key.py and make a variable "api_key = [insert key]"
        game_agent = ChatOpenAI(temperature=0, openai_api_key=api_key) # Agent that will reason over current game
        with open(game_path, "rb") as game_file:
            game_captions = pickle.load(game_file)
        print(game_path, len(game_captions))

        game_index = Chroma.from_texts(["Nothing"],hf)
        game_recency = []
        questions = [None]*len(game_captions) # Array to store questions, no question = None else (q1,q2,q3)
        questions[4] = ("Have you made Diamon", "Can player make Diamond", "What is the person making")

        for i_s, scene in enumerate(game_captions):
            # Game_index for q1,q2; game_recency for time based retrieval for q3 
            game_index.add_texts([scene])
            game_recency.append(scene)

            # If this timestep has an associated question, ask chatgpt the questions
            if questions[i_s]:
                question1, question2, question3 = questions[i_s]
                top_k_q1 = game_index.similarity_search(question1)[:k]
                generated_answer_q1 = chatgpt_questioning(game_agent, top_k_q1, question1)

                top_k_q2 = game_index.similarity_search(question2)[:k]
                generated_answer_q2 = chatgpt_questioning(game_agent, top_k_q2, question2)

                #generated_answer_q3 = chatgpt_questioning(game_agent, game_recency[:k], question3)
                generated_answer_q3 = "AA"

                print(generated_answer_q1, generated_answer_q2, generated_answer_q3)
                # TODO: Get answer from result

        print(game_path, "finish")

if __name__ == "__main__":
    main()
