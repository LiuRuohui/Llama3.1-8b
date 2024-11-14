import traceback
import multiprocessing
from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from Task1 import code_generation
from evaluate_functional_correctness import entry_point

class TimeoutException(Exception):
    pass

def run_code(check_program, entry_point, queue):
    #Execute the code in the child process and put the result in a queue
    local_scope = {}
    try:
        exec(check_program, local_scope)
        if entry_point not in local_scope:
            queue.put(("failed", "Function entry point not found.", "Function entry point not found"))
        else:
            queue.put(("passed", "Execute Successfully", "Execute Successfully"))
    except Exception as e:
        error_message = f"failed: {str(e)}"
        error_message2 = traceback.format_exc()
        queue.put(("error message", error_message, error_message2))
        #print("this is error message\n", error_message)
        traceback.print_exc()

def test_generated_code_with_result(task_id, prompt, generated_code, entry_point, test_code, timeout):
    #Test the generated code and return the execution results and status
    check_program = f"{prompt}\n{generated_code}\n{test_code}\ncheck({entry_point})"
    #print("\nThis is test code\n", test_code, "\nThis is entry point\n", entry_point)
    #print(f"this is{task_id}\n",check_program)
    # Creates a queue for the child process to return results to the main process
    queue = multiprocessing.Queue()
    # Create a child process to execute the code
    process = multiprocessing.Process(target=run_code, args=(check_program, entry_point, queue))
    process.start()
    # Wait for the child process to complete within the specified timeout period
    process.join(timeout)
    # Check whether the child process completed within a timeout
    if process.is_alive():
        process.terminate()  # Timeout terminates the child process
        process.join()  # Wait for the child process to terminate
        print(f"Task ID {task_id} Time-Out")
        return ("timed out", f"Task ID {task_id} execution timed out.", f"Task ID {task_id} execution timed out.")
    # Gets the execution result of the child process
    if not queue.empty():
        result, message, detailed_message = queue.get()
        return result, message, detailed_message
    else:
        return ("failed", "Unknown error: no result from execution.")

if __name__ == "__main__":
    problems = read_problems()
    generated_solutions = []
    count = 0

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        test = problems[p]["test"]
        entrypoint = problems[p]["entry_point"]

        # 生成代码
        code, token0 = code_generation(prompt)
        if code:
            generated_solutions.append({
                "task_id": task_id,
                "prompt": prompt,
                "output": code,
                "completion_token": token0
            })
            print(f"Task ID: {task_id}, Tokens: {token0}")

        # Call test function
        result, message, detailed_message = test_generated_code_with_result(task_id, prompt, code, entrypoint, test, timeout=3.0)
        print("\n\n\n",result,"\n\n\n", message,"\n\n\n", detailed_message)  # Output "passed" or "failed" and error message

        if "passed" in result:
            count += 1

    print("Number of successful cases:", count)
    write_jsonl("TEST.jsonl", generated_solutions)
    final_result = entry_point("TEST.jsonl", k="1", n_workers=4, timeout=3.0)

