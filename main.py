from llm_sdk import Small_LLM_Model

from .function import Function
import json

def big_prompt():
    #bla bla




def main():
    try:
        llm = Small_LLM_Model()

    with open(call_me_maybe/data/input/functions_definition.json, "r", encoding="utf-8") as functions_file:
        load_functions = json.load(functions_file)
        
    with open(call_me_maybe/data/input/function_calling_tests.json) as prompt_file:
        load_calls = json.load(prompt_file)

    functions_list = []
        for idx, data in enumerate(load_functions):
            function = Function.create_from_dict(data, idx)
            function_list.append(function)
            
    for call in load_calls:
        prompt = call.get("prompt")
        full_prompt = big_prompt(prompt, function_list)
        tokens = llm.encode(full_prompt)
        


    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 

    return 