import json
import requests
import os
from agent import BasicAgent

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_random():
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/random-question"

    file_path = "questions.json"
    question_data = {}

    if os.path.exists(file_path):
        print("Questions cache file exists!")
        with open(file_path, "r") as json_file:
            all_questions = json.load(json_file)
            question_data = all_questions[0]
    # else:
    #     print(f"Fetching questions from: {questions_url}")
    #     try:
    #         response = requests.get(questions_url, timeout=15)
    #         response.raise_for_status()
    #         question_data = response.json()
    #         if not question_data:
    #             print("Fetched questions list is empty.")
    #             return "Fetched questions list is empty or invalid format.", None
    #         print(f"Fetched {len(question_data)} questions.")
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error fetching questions: {e}")
    #         return f"Error fetching questions: {e}", None
    #     except requests.exceptions.JSONDecodeError as e:
    #         print(f"Error decoding JSON response from questions endpoint: {e}")
    #         print(f"Response text: {response.text[:500]}")
    #         return f"Error decoding server response for questions: {e}", None
    #     except Exception as e:
    #         print(f"An unexpected error occurred fetching questions: {e}")
    #         return f"An unexpected error occurred fetching questions: {e}", None
    #     with open(file_path, "w") as json_file:
    #         json.dump(question_data, json_file, indent=4)

    task_id = question_data.get("task_id")
    question_text = question_data.get("question")
    submitted_answer = {}
    try:
        submitted_answer = agent(question_text)
    except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")

    print(f"The questions was:  {question_text}\nThe answer given was:  {submitted_answer}")

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    run_random()