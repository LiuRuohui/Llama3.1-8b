import time
from baseline import read_problems, write_jsonl
from evaluate_functional_correctness import entry_point
from openai import OpenAI
import openai

# API Keys List
API_KEYS = [
    "bab5a926-5245-4843-a03d-d98b57a0c644",
    "31a55a95-3f5c-483b-9a35-5fa473a6006a",
    "216ca667-90ec-490b-978a-476ab39bcb79",
    "6c1efdc4-fcb7-45b8-962b-aec70649b27f",
    "1a6214e2-084a-4987-bb4e-48f770b8e068",
]
api_key_index = 0

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])


# category list
categories = [
    "Basic Algorithm Problems",
    "Mathematical Problems",
    "Data Structure Related Problems",
    "String Operations",
    "Logical Reasoning and Conditional Statements",
    "Complexity Analysis",
    "Specific Function Implementations",
    "Debugging and Code Fixing"
]

def clean_the_wrap(code):
    # Remove ``` markers and the word 'python'
    start_index = code.find('```')
    if start_index!= -1:
        end_index = code.find('```', start_index + 3)
        if end_index!= -1:
            code = code[start_index + 3:end_index].replace("python", "")
    return code.strip()

# Initialize categorized results
categorized_results = {category: [] for category in categories}

def questionType_generation(prompt, retries=5, delay=1):
    global api_key_index

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "give me the type of the problem based on your knowledge. Remember, you could only select one from the below list:\n Basic Algorithm Problems, Mathematical Problems, Data Structure Related Problems, String Operations, Logical Reasoning and Conditional Statements, Complexity Analysis, Specific Function Implementations, Debugging and Code Fixing.\n If you can't specify it, you can select two or three of them from the list."},
                    {"role": "user", "content": f"You only need to output the type of the question above the list without any other statements\n{prompt}"}
                ],
                stream=True,
                stream_options={"include_usage": True}
            )

            full_response = ""
            tokens = 0
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        full_response += delta.content
                if chunk.usage:
                    tokens += chunk.usage.completion_tokens

            return full_response, tokens, f"give me the type of the problem based on your knowledge. Remember, you could only select one from the below list:\n Basic Algorithm Problems, Mathematical Problems, Data Structure Related Problems, String Operations, Logical Reasoning and Conditional Statements, Complexity Analysis, Specific Function Implementations, Debugging and Code Fixing.\n If you can't specify it, you can select two or three of them from the list.\n User:{prompt}"  # Return full response and tokens

        except openai.RateLimitError:
            print("Rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)
            client.api_key = API_KEYS[api_key_index]
            time.sleep(delay)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(delay)

    return None, 0

def design_prompt_simple(questionType):
    specific_prompts = []
    if "Basic Algorithm Problems" in questionType:
        specific_prompts.append("This is a basic algorithm problem")
    if "Mathematical Problems" in questionType:
        specific_prompts.append("This is a mathematical problem.")
    if "Data Structure Related Problems" in questionType:
        specific_prompts.append("This is a data structure related problem.")
    if "String Operations" in questionType:
        specific_prompts.append("This is a string operation related problem.")
    if "Logical Reasoning and Conditional Statements" in questionType:
        specific_prompts.append("This is a logical reasoning and conditional statements related problems.")
    if "Complexity Analysis" in questionType:
        specific_prompts.append("This is complexity related problem.")
    if "Specific Function Implementations" in questionType:
        specific_prompts.append("This is specific function implementation.")
    if "Debugging and Code Fixing" in questionType:
        specific_prompts.append("This is debugging and code fixing related problem.")
    # 每行输出一个 specific prompt
    final_prompt = "\n".join(specific_prompts)
    return final_prompt
def design_prompt_medium(questionType):
    specific_prompts = []
    if "Basic Algorithm Problems" in questionType:
        specific_prompts.append("Solve this problem using basic algorithms. Consider using well - known sorting or searching algorithms if applicable. Ensure the code is efficient by analyzing its complexity and easy to understand with proper comments.")
    if "Mathematical Problems" in questionType:
        specific_prompts.append("Provide a detailed mathematical solution to the following problem. Explain each step clearly, showing the reasoning behind the mathematical operations and formulas used.")
    if "Data Structure Related Problems" in questionType:
        specific_prompts.append("Design an efficient solution using appropriate data structures. Think about the nature of the data and operations required, and choose data structures like arrays, linked lists, or trees accordingly.")
    if "String Operations" in questionType:
        specific_prompts.append("Solve this string manipulation task with clarity and correctness. Pay attention to details such as string length, character encoding, and possible edge cases like empty strings.")
    if "Logical Reasoning and Conditional Statements" in questionType:
        specific_prompts.append("Apply logical reasoning to address the following problem. Break down the problem into smaller logical parts and use conditional statements effectively to handle different scenarios.")
    if "Complexity Analysis" in questionType:
        specific_prompts.append("Analyze the time and space complexity of the solution after implementing the code. Use appropriate methods like Big - O notation to accurately represent the complexity and consider how the complexity might change with different input sizes.")
    if "Specific Function Implementations" in questionType:
        specific_prompts.append("Implement the specific function described in the prompt with proper error handling. Anticipate possible errors such as invalid inputs and handle them gracefully to prevent the program from crashing.")
    if "Debugging and Code Fixing" in questionType:
        specific_prompts.append("Debug and fix the given code snippet to ensure it works correctly. Use debugging tools and techniques like print statements or debugging environments to identify and resolve issues like logical errors or runtime errors.")
    # 每行输出一个 specific prompt
    final_prompt = "\n".join(specific_prompts)
    return final_prompt
def design_prompt_complex(questionType):
    specific_prompts = []

    if "Basic Algorithm Problems" in questionType:
        specific_prompts.append(
            "Basic Algorithm Problems focus on solving computational challenges using foundational techniques such as sorting, searching, and recursion. "
            "For example, sorting algorithms like QuickSort and MergeSort offer efficient ways to handle ordered data, while searching techniques like binary search optimize lookups in sorted arrays. "
            "Dynamic programming methods like memoization and tabulation can be used for problems with overlapping subproblems, such as the Fibonacci sequence. "
            "Ensure that the solution adheres to constraints, considers edge cases (e.g., empty inputs or large datasets), and optimizes both time and space complexity for scalability."
        )
    if "Mathematical Problems" in questionType:
        specific_prompts.append(
            "Mathematical Problems require translating abstract numerical or algebraic concepts into computational solutions. "
            "Tasks may involve operations such as matrix manipulation, number theory, statistical calculations, or optimization problems. "
            "For instance, implementing numerical methods like Newton-Raphson for root finding, or solving linear equations using Gaussian elimination, requires a careful balance of precision and computational efficiency. "
            "Explain the mathematical theory behind the solution, demonstrate any assumptions, and ensure the implementation accounts for floating-point errors and edge cases, such as division by zero or invalid inputs."
        )
    if "Data Structure Related Problems" in questionType:
        specific_prompts.append(
            "Data Structure Related Problems revolve around leveraging specialized data organizations, such as stacks, queues, linked lists, trees, graphs, or hash maps, to meet specific computational requirements. "
            "Key considerations include trade-offs between insertion, deletion, and search times, as well as memory usage. "
            "For instance, using a heap for priority queues optimizes access to the smallest or largest element, while a trie efficiently handles prefix-based queries. "
            "Design the solution by first analyzing the problem's constraints and expected operations, then selecting a data structure that minimizes computational overhead and maximizes clarity."
        )
    if "String Operations" in questionType:
        specific_prompts.append(
            "String Operations involve manipulating textual data through tasks like pattern matching, substring extraction, or text formatting. "
            "Efficient solutions often employ techniques such as the KMP (Knuth-Morris-Pratt) algorithm for pattern matching, character frequency tables for anagram detection, or sliding window approaches for substring problems. "
            "Incorporate robust error handling to manage edge cases like empty strings, special characters, or language-specific encodings (e.g., UTF-8). Clearly document how the approach scales with input length and ensure that regex or other tools are used only when appropriate for simplicity and performance."
        )
    if "Logical Reasoning and Conditional Statements" in questionType:
        specific_prompts.append(
            "Logical Reasoning and Conditional Statements require decomposing complex conditions into executable logic, often involving constructs such as if-else, switch statements, or boolean expressions. "
            "For example, consider a nested conditional logic problem requiring evaluation of multiple rules, where clear and well-commented decision trees or boolean algebra simplifications can aid in reducing complexity. "
            "Explain the reasoning behind each logical step, validate edge cases (e.g., boundary values in ranges), and ensure code readability through concise comments and modular structure."
        )
    if "Complexity Analysis" in questionType:
        specific_prompts.append(
            "Complexity Analysis is crucial for evaluating the efficiency of a solution. It involves analyzing both time complexity (e.g., O(n), O(log n)) and space complexity based on the algorithm's design. "
            "Explain the computational costs for each critical step of the implementation, identifying potential bottlenecks or areas for optimization. "
            "Use examples to compare theoretical predictions with empirical results, such as analyzing runtime for inputs of varying sizes, and discuss whether improvements like parallelism, caching, or data structure adjustments could further optimize performance."
        )
    if "Specific Function Implementations" in questionType:
        specific_prompts.append(
            "Specific Function Implementations require constructing modular and reusable code for well-defined tasks. "
            "For example, a function to calculate the median of a dataset should handle both odd and even input sizes and ensure correctness through sorting or heap-based techniques. "
            "Prioritize parameter validation, include docstrings describing the function's inputs, outputs, and edge cases, and implement robust error handling to prevent unexpected failures (e.g., handling None or invalid input types)."
        )
    if "Debugging and Code Fixing" in questionType:
        specific_prompts.append(
            "Debugging and Code Fixing entail identifying and resolving errors in existing code. Common techniques include using print statements or logging for traceability, employing debuggers for stepwise execution, and writing targeted unit tests to isolate faulty behavior. "
            "Explain how each identified bug affects program logic and provide a clear rationale for the fix. Additionally, ensure the corrected code adheres to best practices, such as proper variable naming, code comments, and compliance with style guides (e.g., PEP 8 in Python)."
        )

    # Create the final prompt with each specific description on a new line for clarity
    final_prompt = "\n\n".join(specific_prompts)
    return final_prompt


def code_generation(prompt, questionType, retries=5, delay=1):
    global api_key_index

    #final_prompt = design_prompt_simple(questionType)
    final_prompt = design_prompt_medium(questionType)
    #final_prompt = design_prompt_complex(questionType)
    #print("This is final prompt\n", final_prompt)

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": f"\n{final_prompt}\n"},
                    {"role": "user", "content": f"\n{prompt}\n"}
                ],
                stream=True,
                stream_options={"include_usage": True}
            )

            full_response = ""
            tokens = 0
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        full_response += delta.content
                if chunk.usage:
                    tokens += chunk.usage.completion_tokens

            return full_response, tokens, f"System:Generate the code based on specific question type .\n User:\n{final_prompt}\n{prompt}\n"  # Return full response and tokens

        except openai.RateLimitError:
            print("Rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)
            client.api_key = API_KEYS[api_key_index]
            time.sleep(delay)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(delay)

    return None, 0


if __name__ == '__main__':
    problems = read_problems()
    questionTypes = []
    task_ids = []
    prompts = []
    generated_solutions = []

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        prompts.append(prompt)
        task_ids.append(task_id)

    total_time = 0
    total_tokens = 0

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        Type_completion, token_count1, prompt_given = questionType_generation(prompt)
        #print("\n\n", task_id, "\n", Type_completion)
        total_tokens += token_count1
        completion, token_count2, prompt_given = code_generation(prompt, Type_completion)
        #print("code\n", clean_the_wrap(completion))
        elapsed_time = time.time() - start_time

        total_time += elapsed_time
        total_tokens += token_count2

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "prompt": prompt,
                "output": clean_the_wrap(completion),
                "elapsed_time": elapsed_time,
                "total_token": token_count1 + token_count2
            })

            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count1 + token_count2}")


    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    # Save results
    # simple prompt
    #write_jsonl("prompt_complexity_simple.jsonl", generated_solutions)
    #entry_point("prompt_complexity_simple.jsonl", k="1", n_workers=4, timeout=5.0)
    # medium prompt
    write_jsonl("prompt_complexity_medium.jsonl", generated_solutions)
    entry_point("prompt_complexity_medium.jsonl", k="1", n_workers=4, timeout=5.0)
    # complex prompt
    #write_jsonl("prompt_complexity_complex.jsonl", generated_solutions)
    #entry_point("prompt_complexity_complex.jsonl", k="1", n_workers=4, timeout=5.0)
