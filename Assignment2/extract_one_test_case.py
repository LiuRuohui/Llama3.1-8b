

import re
import random
from openai import OpenAI
import openai
from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from Task1 import code_generation
from test_single import test_generated_code_with_result
from evaluate_functional_correctness import entry_point

def extract_one_test_case(test_code):
    # Matches the content from the first assert to the second assert
    pattern = r'(assert.*?)(?=\n\s*assert|$)'
    matches = re.findall(pattern, test_code, re.DOTALL)
    # Randomly select one match and return
    if matches:
        return "def check(candidate):\n    " + random.choice(matches).strip()
    else:
        return None
def extract_from_prompt(prompt):
    # Matches lines starting with >>> to extract test cases
    pattern = r'>>> (.*?)(?:\n|$)(.*?)\n'
    matches = re.findall(pattern, prompt, re.DOTALL)

    # Select one test case at random and format it
    if matches:
        chosen_case = random.choice(matches)
        test_input = chosen_case[0].strip()    # >>> line content
        expected_output = chosen_case[1].strip()  # next line with expected output
        # Construct the test case
        return f"def check(candidate):\n    assert candidate({test_input}) == {expected_output}"
    else:
        return None


if __name__ == "__main__":
    problems = read_problems()
    generated_solutions = []
    count = 0

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        test = problems[p]["test"]
        entrypoint = problems[p]["entry_point"]

        code, token0 = code_generation(prompt)
        if code:
            generated_solutions.append({
                "task_id": task_id,
                "prompt": prompt,
                "output": code,
                "completion_token": token0
            })
            print(f"Task ID: {task_id}, Tokens: {token0}")
        test_case = extract_one_test_case(test)
        #test_case = extract_from_prompt(prompt)
        print(test_case)
        #print(test_case1)
        #print(f"{task_id}\n{prompt}")

        result, message, *_ = test_generated_code_with_result(task_id, prompt, code, entrypoint, test_case, timeout=5.0)
        print(task_id, "\n", result, message)

        if "passed" in result:
            count += 1
    print("Number of successful cases:", count)
    write_jsonl("Test_one_case.jsonl", generated_solutions)
    final_result = entry_point("Test_one_case.jsonl", k="1", n_workers=4, timeout=3.0)


