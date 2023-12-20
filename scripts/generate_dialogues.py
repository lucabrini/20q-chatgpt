import csv
import os
import time
import random
import re 
import argparse
from functools import wraps
import json
from dotenv import load_dotenv
import openai
from tqdm import tqdm

from uncertainty.custom_model import LLModelWrapper

csv_header = [
    "dialogue_id", # enumeration of games dialogues
    "intra_dialogue_id" # question/answer index inside dialogue with id = dialogue_id
    "target" # item assigned to user
    "question", # question made by the guesser
    "answer", # response from the user/oracle
    "question_confidence",
    "question_observed_consistency",
    "question_self_reflection",
    "answer_confidence",
    "answer_observed_consistency",
    "answer_self_reflection",
    "question_time",
    "answer_time"
]

def retry_on_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except openai.error.RateLimitError:
                print("Rate limit reached. Waiting for 10 seconds...")
                time.sleep(10)
    return wrapper

def get_lists_of_candidates(constrast_sets):
  list_and_target = {}
  count_ = 0
  for contrast_set in constrast_sets.values():
    list_and_target[count_] = {'candidates':contrast_set['items'], 'target':contrast_set['target']}
    count_ += 1
  return list_and_target

def openai_call(model: LLModelWrapper, conversation, oracle=False):
    if oracle:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=conversation,
            temperature=0.1,
        )
    else:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=conversation,
        )
    return {'role': response.choices[0].message.role, 'content': response.choices[0].message.content}

def get_prompts(candidates, target, stepwise=False):
    if stepwise:
        questioner = ([{'role': "system", 'content': "You are playing an interactive game with the user, who is assigned "\
                                                        "an item from a list of candidates. Ask as few questions as possible to identify the item, "\
                                                        "making only one question at each turn.\n"\
                                                        "\nThe user can only respond with 'yes' or 'no'.\n"\
                                                        "Format your output in the following way:\n"\
                                                        "CANDIDATES: item, item, item, item ...\n"\
                                                        "QUESTION: text of the question"},
                            {'role': "user", 'content': f"This is the list of candidates: {candidates}."}])
    else:
        questioner = ([{'role': "system", 'content': "You are playing an interactive game with the user, who is assigned "\
                                                        "an item from a list of candidates. Ask as few questions as possible to identify the item, "\
                                                        "making only one question at each turn.\n"\
                                                        "\nThe user can only respond with 'yes' or 'no'."},
                            {'role': "user", 'content': f"This is the list of candidates: {candidates}."}])
    
    oracle = ([{'role': "system", 'content': "You are playing an interactive game with the user, in which you are assigned one item from a list "\
                                                "of candidates."\
                                                "\nThe user will have to guess which one it is by asking yes/no questions, and "\
                                                "you have to stricly respond to each question only with 'yes' or 'no'."\
                                                "\nIf the user correctly guesses exactly your assigned item, respond with 'Yes! That's correct.'."\
                                                f"\nThe item assigned to you is {target}."}])
    return questioner, oracle


def generate_dialogues_openai(model: LLModelWrapper, target_list_candidates, game_set, num_candidates, data_path=f"./data" ):
    
    with open(data_path + f"/generation/{game_set}/dialogues.csv", 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(csv_header)
    
    if os.path.exists(data_path + f"/generation/{game_set}/dialogues.txt"):
        with open(data_path + f"/generation/{game_set}/dialogues.txt", "r") as f:
            dialogues_raw_txt = f.read()
            num_dialogues = len(dialogues_raw_txt.split("******************"))
            target_list_candidates = {key: target_list_candidates[key] for key in target_list_candidates.keys() if int(key) >= (num_dialogues - 1)}

    else:
        if not os.path.exists(data_path + f"/generation/{game_set}"):
            os.mkdir(data_path + f"/generation/{game_set}")
        num_dialogues = 0

    stepwise = True if "stepwise" in game_set else False

    for dialogue_id, value in target_list_candidates.items():
        print(dialogue_id, value)
        successful = False
        while not successful:

            dialogue = []

            target = value['target']

            # print("******************")
            dialogue.append("******************")
            # print(f"target = {target}")
            dialogue.append(f"target = {target}")

            # Initial prompts. Game rules
            questioner, oracle = get_prompts(", ".join(value['candidates']), target, stepwise=stepwise)

            # print('answerer: {}\t'.format(questioner[-1]['content'].strip()))
            dialogue.append('answerer: {}'.format(questioner[-1]['content'].strip()))

            oracle_output = {"content" : ""}

            for intra_dialogue_id in range(20):

                # Generating new question
                time_start = time.time()
                questioner_output, question_uncertainty_metrics = model.ask(
                    question="This is the current dialogue: " + "\n".join(dialogue[2:]) ,
                    message_history=[questioner[0]] # Task prompt
                )
                time_end = time.time()
                question_time = time_end - time_start
                questioner.append({'role': 'assistant', 'content': re.sub(r"\n\n*", " ", questioner_output)})
                try:
                    processed_questioner_output = questioner_output.split("QUESTION:")[1].strip()
                except IndexError:
                    processed_questioner_output = questioner_output
                
                # Appending new question
                oracle.append({'role': 'user', 'content': processed_questioner_output})
                generated_question = questioner[-1]['content'].strip()
                
                # Dialogue
                # print('questioner: {}\t'.format(generated_question))
                dialogue.append('questioner: {}'.format(generated_question))

                # Generating new question's answer
                time_start = time.time()
                oracle_output, answer_uncertainty_metrics = model.ask(
                    question="This is the current dialogue: " + "\n".join(dialogue[2:]) ,
                    message_history=[oracle[0]], # Task prompt
                    temperature=0.1,
                )
                time_end = time.time()
                answer_time = time_end - time_start
                
                # Appending new question's answer
                questioner.append({'role': 'user', 'content': re.sub("\n", " ", oracle_output)})
                oracle.append({'role': 'assistant', 'content': oracle_output})
                generated_answer = questioner[-1]['content'].strip()
                
                # Dialogue
                # print('answerer: {}\t'.format(generated_answer))
                dialogue.append('answerer: {}'.format(generated_answer))
                
                with open(data_path + f"/generation/{game_set}/dialogues.csv", 'a') as f:
                    write = csv.writer(f)
                    write.writerow([
                        dialogue_id,
                        intra_dialogue_id,
                        target, 
                        generated_question,
                        generated_answer,
                        round(question_uncertainty_metrics["confidence"], 5),
                        round(question_uncertainty_metrics["observed_consistency"], 5),
                        round(question_uncertainty_metrics["self_reported_certainty"], 5),
                        round(answer_uncertainty_metrics["confidence"], 5),
                        round(answer_uncertainty_metrics["observed_consistency"], 5),
                        round(answer_uncertainty_metrics["self_reported_certainty"], 5),
                        round(question_time*1000),
                        round(answer_time*1000)
                    ])

                if "correct" in oracle_output.lower() and "yes" in oracle_output.lower():
                    with open(data_path + f"/generation/{game_set}/dialogues.txt", "a") as f:
                        for line in dialogue:
                            f.write(f"{line}\n")
                    successful = True
                    break


