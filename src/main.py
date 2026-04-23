from llm_sdk.llm_sdk import Small_LLM_Model
from math import inf
from .function import Function
import json

def big_prompt(prompt_str, functions_list) -> str:
    prompt = "Available functions:\n"
    prompt += full_fn_list(functions_list)
    prompt += "Return name of the function:\n"
    prompt += "For example: \n"
    prompt += "Input: What is the sum of 2 and 3?\n"
    prompt += "Output: fn_add_numbers \n"
    prompt += "Input: "
    prompt += prompt_str
    prompt += "\nOutput: "
    return prompt

def prompt_for_param(function, big_prompt):
    prompt = ""


def get_function_from_name(name, functions_list):
    for fn in functions_list:
        if name == fn.name:
            return fn
    return functions_list[0]


def full_fn_list(functions):
    full_str = ""
    for function in functions:
        full_str += function.fn_to_prompt()
        full_str += "\n"
    return full_str

def all_functions_names(functions):
    all_names = []
    for function in functions:
        all_names.append(function.name)
    return all_names
       

def main():
    #try:
        
    llm = Small_LLM_Model(device="cpu")
    with open("data/input/functions_definition.json", "r", encoding="utf-8") as functions_file:
        load_functions = json.load(functions_file)
        
    with open("data/input/function_calling_tests.json",  "r", encoding="utf-8") as prompt_file:
        load_calls = json.load(prompt_file)
    functions_list = []
    for idx, data in enumerate(load_functions):
        function = Function.create_from_dict(data, idx)
        functions_list.append(function)
            
    for call in load_calls:
        prompt = call.get("prompt")
        full_prompt = big_prompt(prompt, functions_list)
        tokens = llm.encode(full_prompt).tolist()[0]
        generated_tokens = []
        generated_name = ""
        allowed_names = all_functions_names(functions_list)
        print(prompt)
        for i in range(10):
            #print(llm.decode(tokens + generated_tokens))
            logits = llm.get_logits_from_input_ids(tokens + generated_tokens)
            max_logit = float(-inf)
            max_idx = -inf
            masked_logits = len(logits) * [-inf]
            allowed_tokens = [llm.encode(name).tolist()[0] for name in allowed_names]
            for tokens_list in allowed_tokens:
                if len(tokens_list) > i:
                    masked_logits[tokens_list[i]] = logits[tokens_list[i]]
                #for token in tokens_list:
                    #masked_logits[token] = logits[token]
            #print(allowed_tokens)
            for idx, logit in enumerate(masked_logits):
                if logit > max_logit:
                    max_logit = logit
                    max_idx = idx
            generated_tokens.append(max_idx)
            generated_name = str(llm.decode(generated_tokens)).strip()
            #print("1", generated_name)
            new_allowed_names = []
            for name in allowed_names:
                if name.startswith(generated_name):
                    new_allowed_names.append(name)
            if len(new_allowed_names) == 1:
                generated_name = new_allowed_names[0]
            #print("2", generated_name)
            
            allowed_names = new_allowed_names

            if generated_name in allowed_names:
                break
            
            
            #print(generated_tokens)
        selected_function = get_function_from_name(generated_name, functions_list)
        
        print(selected_function.to_json())
        #last_prompt = prompt_for_param(prompt, generated_name)  



         
        #print(tokens)
    
    #except Exception as exc:
        #print(f"Unexpected error: {exc}")
       # return 

    return 