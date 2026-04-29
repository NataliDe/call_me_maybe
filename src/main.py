from llm_sdk.llm_sdk import Small_LLM_Model
from math import inf
from .function import Function
import json
import os

def big_prompt(prompt_str, functions_list) -> str:
    prompt = "\n\nAvailable functions:\n"
    prompt += full_fn_list(functions_list)
    prompt += "Return name of the function:\n"
    prompt += "For example: \n"
    prompt += "Input: What is the sum of 2 and 3?\n"
    prompt += "Output: fn_add_numbers \n"
    prompt += "Input: "
    prompt += prompt_str
    prompt += "\nOutput: "
    return prompt

def prompt_for_param(base_prompt, json_part, function):
    prompt = "\n\nFor this function:\n"
    prompt += function.fn_to_prompt()
    prompt += "\nReturn ONLY valid JSON in the format:\n"
    prompt += '{"prompt": "...", "name": "...", "parameters": { ... } }\n'
    prompt += 'for example:\n'
    prompt += 'User: "What is the sum of 2 and 3"\n'
    prompt += ('Output: {"prompt": "What is the sum of 2 and 3", '
               '"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}')


    prompt += "\n\nUser: "
    prompt += base_prompt
    prompt += "\n"
    prompt += "Output: "
    prompt += json_part
    
    return prompt







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

def strip_value(val):
    value = str(val)
    value = value.split('\n')[0]
    value = value.strip('"')
    value = value.strip("'")
    return value.strip()

def convert_value(val, param_type):
    value = strip_value(val)
    if param_type == "number":
        return float(value)
    if param_type == "boolean":
        return bool(value)
    return value

def is_float(gener_text):
    try:
        int(gener_text)
        return False
    except Exception:
        try:
            float(gener_text)
            return True
        except Exception:
            return False


def argument_is_finished(generated_text, param_type):
    if param_type == "number":
        return is_float(generated_text)
    if param_type == "boolean":
        return generated_text in ["true", "false"]
    return any(c in generated_text for c in ['}', '"'])



def generate_value(llm, full_prompt_paramaateres, param_type):
    tokens = llm.encode(full_prompt_paramaateres).tolist()[0]

    max_tokens = 25
    generated_tokens = []


    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(tokens + generated_tokens) 
        #print(llm.decode(tokens + generated_tokens))

        max_logit = float(-inf)
        max_idx = -inf
        for idx, logit in enumerate(logits):
            if logit > max_logit:
                max_logit = logit
                max_idx = idx 

        
        generated_tokens.append(max_idx)
        generated_text = str(llm.decode(generated_tokens))
        print(generated_text)


        if argument_is_finished(generated_text, param_type):
            break
    if param_type == 'string':
        return(generated_text.split('"')[0])

    
    

 
    return (generated_text)
       

def main():
    try:
        llm = Small_LLM_Model(device="cpu")

        with open("data/input/functions_definition.json", "r", encoding="utf-8") as functions_file:
            load_functions = json.load(functions_file)
        with open("data/input/function_calling_tests.json",  "r", encoding="utf-8") as prompt_file:
            load_calls = json.load(prompt_file)

        functions_list = []
        for idx, data in enumerate(load_functions):
            function = Function.create_from_dict(data, idx)
            functions_list.append(function)

        results = []

        for call in load_calls:
            prompt = call.get("prompt")
            full_prompt = big_prompt(prompt, functions_list)
            tokens = llm.encode(full_prompt).tolist()[0]
            generated_tokens = []
            generated_name = ""
            allowed_names = all_functions_names(functions_list)

            for i in range(10):
                logits = llm.get_logits_from_input_ids(tokens + generated_tokens)
                masked_logits = len(logits) * [-inf]

                allowed_tokens = [llm.encode(name).tolist()[0] for name in allowed_names]


                for tokens_list in allowed_tokens:
                    if len(tokens_list) > i:
                        masked_logits[tokens_list[i]] = logits[tokens_list[i]]
                    #for token in tokens_list:
                        #masked_logits[token] = logits[token]
                #print(allowed_tokens)

                max_logit = float(-inf)
                max_idx = -inf
                for idx, logit in enumerate(masked_logits):
                    if logit > max_logit:
                        max_logit = logit
                        max_idx = idx

                generated_tokens.append(max_idx)
                generated_name = str(llm.decode(generated_tokens)).strip()

                new_allowed_names = []
                for name in allowed_names:
                    if name.startswith(generated_name):
                        new_allowed_names.append(name)
                if len(new_allowed_names) == 1:
                    generated_name = new_allowed_names[0]
                allowed_names = new_allowed_names

                if generated_name in allowed_names:
                    break

            selected_function = get_function_from_name(generated_name, functions_list)
            #print(selected_function.to_json(prompt))


            for index, key in enumerate(selected_function.parameters):
                #print(key)
                json_part = selected_function.to_json_parts(prompt, index)
                full_prompt_paramaateres = prompt_for_param(prompt, json_part, selected_function)
                print(full_prompt_paramaateres)

                param_type = selected_function.parameters_type.get(key, "string")
                generated_value = generate_value(llm, full_prompt_paramaateres, param_type)
                #print(generated_value)


                
                selected_function.parameters[key] = convert_value(generated_value, param_type)
                #print(selected_function.parameters[key])
            results.append(selected_function.to_dict(prompt))
            print (results)

        os.makedirs("data/output", exist_ok=True)

        with open("data/output/functions_calling_results.json", "w", encoding="utf-8") as output_file:
                json.dump(results, output_file, indent=2)


                

            
    except Exception as exc:
        print(f"Unexpected error: {exc}")
    return 