import os
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import pickle
from glob import glob
from api_key import api_key

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
k = 1

def chatgpt_questioning(agent, top_k_history, question, dialogue, q3=False):

    with open('prompt.txt', 'r') as file:
        sys_msg = file.read()

    #hmn_msg1 = f"Top 2 Visual Description:\n{top_k_history}\n Chat Dialogue: {dialogue}\nQuestion:{question}\nOptions:[Yes, No, Maybe]\n Answer: Let's think step by step."
    if not q3:
        #hmn_msg1 = f"Visual Description:\n{top_k_history}\n Chat Log:\n {dialogue} \nQuestion:{question}\nOptions:[YES, NO, MAYBE]\nAnswer: Let's think step by step."
        hmn_msg1 = f"Chat Log:\n {dialogue} \nQuestion:{question}\nOptions:[YES, NO, MAYBE]\nAnswer: Let's think step by step."
    else:
        #hmn_msg1 = f"Visual Description:\n{top_k_history}\n Chat Log:\n {dialogue} \nQuestion:{question}\nAnswer: Let's think step by step."
        hmn_msg1 = f"Chat Log:\n {dialogue} \nQuestion:{question}\nAnswer: Let's think step by step."
    print(hmn_msg1)
    print()
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=hmn_msg1)]
    out_1 = agent(messages)
    return out_1

def main():
    first = 0
    for game_path in glob("image2textMindcraft/*.pkl"):
        game_captions = [] # Store captions for each frame
        questions = [] # Store captions for each frame
        dialogues = [] # Store captions for each frame
        descriptions = []

        # IMPORTANT! REPLACE WITH API KEY!
        # MAKE A FILE CALLED api_key.py and make a variable "api_key = [insert key]"
        game_agent = ChatOpenAI(temperature=0, openai_api_key=api_key, max_tokens=214) # Agent that will reason over current game
        game_number = game_path.split("/")[-1].split(".")[0]
        with open(f"image2CLIPquestions/{game_number}.pkl", "rb") as questions_file:
            questions = pickle.load(questions_file)

        with open(f"image2CLIPdialogue/{game_number}.pkl", "rb") as dialogue_file:
            dialogues = pickle.load(dialogue_file)

        with open(f"image2CLIP/{game_number}.pkl", "rb") as descriptions_file:
            descriptions = pickle.load(descriptions_file)
        print(game_number)
        if first == 0:
            first = 232
            continue
        start_ts = 0
        end_ts = 0
        if dialogues:
            start_ts = dialogues[0][0]
            start_ts = min(start_ts, questions[0][0])
            end_ts = max(dialogues[-1][0], questions[-1][0])
        else:
            start_ts = questions[0][0]
            end_ts = questions[-1][0]
        total_seconds = end_ts - start_ts
        frames_per_s = len(descriptions)*60//total_seconds
        frame_rate = None

        if frames_per_s < 10:
            frame_rate = 6
        elif frames_per_s < 20:
            frame_rate = 12
        elif frames_per_s < 45:
            frame_rate = 30
        else:
            frame_rate = 60

        #game_index = Chroma.from_texts(["Nothing"],hf)
        game_recency = []
        ts = 0
        curr_q_ts = 0
        curr_dialogue = ""
        curr_d_ts = 0
        acc = 0
        acc1 = 0
        acc2 = 0
        acc3 = 0

        for i_s, scene in enumerate(descriptions):
            question = None
            ts += 60//frame_rate


            # Game_index for q1,q2; game_recency for time based retrieval for q3 
            #game_index.add_texts([scene])
            game_recency.append(scene)


            # Last dialogue before the 60 frames we have seen so far
            while curr_d_ts < len(dialogues) and dialogues[curr_d_ts][0] <= ts:
                curr_dialogue += dialogues[curr_d_ts][1]
                curr_d_ts += 1 

            # If this timestep has an associated question, ask chatgpt the questions
            if (curr_q_ts+1)*75 <= ts:
                question = questions[curr_q_ts]
                curr_q_ts += 1

            if question:
                print(question)
                question[2][0] = "Has Player 1 made "+question[2][0].split(" ")[-3] if int(question[2][0].split(" ")[0] == 'Have')==0 else "Has Player 2 made "+question[2][0].split(" ")[-3]
                question[2][1] = "Does Player 1 know how to make "+question[2][1].split(" ")[-1] if int(question[2][1].split(" ")[2] == 'know')==0 else \
                        "Does Player 2 know how to make "+question[2][1].split(" ")[-1]
                question[2][2] = "What is Player 1 currently making?" if int(question[2][2].split(" ")[1] == 'are')==0 else "What is Player 2 currently making?"
                print(question)
                question1, question2, question3 = question[2]
                answers = question[-1]

                #top_k_q1 = game_index.similarity_search(question1)[:k]
                generated_answer_q1 = chatgpt_questioning(game_agent, game_recency[-k:], question1.replace('_',' '), curr_dialogue)
                print(question[-1][0], generated_answer_q1)
                ans = input("Is the answer correct? 1: Yes 2: No\n")
                if int(ans) == 1:
                    print("Correct answer registered.")
                    print()
                    acc += 1
                    acc1 += 1

                #top_k_q2 = game_index.similarity_search(question2)[:k]
                generated_answer_q2 = chatgpt_questioning(game_agent, game_recency[-k:], question2, curr_dialogue)
                print(question[-1][1], generated_answer_q2)
                ans = input("Is the answer correct? 1: Yes 2: No\n")
                if int(ans) == 1:
                    print("Correct answer registered.")
                    print()
                    acc += 1
                    acc2 += 1

                generated_answer_q3 = chatgpt_questioning(game_agent, game_recency[:k], question3, curr_dialogue, True)
                print(question[-1][2], generated_answer_q3)
                ans = input("Is the answer correct? 1: Yes 2: No\n")
                if int(ans) == 1:
                    print("Correct answer registered.")
                    print()
                    acc += 1
                    acc3 += 1

        print(game_path, f"Accuracy was {acc/(len(questions)*3)}. EXP1: {acc1/len(questions)} EXP2: {acc2/len(questions)} EXP3: {acc3/len(questions)}", "finish")
        break

if __name__ == "__main__":
    main()
