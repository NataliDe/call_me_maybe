"""Main generation pipeline."""

import json
import re
from math import inf
from typing import Any

from llm_sdk import Small_LLM_Model

from .function import Function
from .io_utils import load_json_file, parse_arguments, save_results


def big_prompt(prompt_str: str, functions_list: list[Function]) -> str:
    """Build prompt for function-name generation."""
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


def prompt_for_param(
    base_prompt: str,
    json_part: str,
    function: Function,
) -> str:
    """Build prompt for one parameter value."""
    prompt = "\n\nFor this function:\n"
    prompt += function.fn_to_prompt()
    prompt += "\nReturn ONLY the next parameter value."
    prompt += "\nDo not explain anything.\n"
    prompt += "\nRules:\n"
    prompt += "- For number parameters, "
    prompt += "return only the number from the user request.\n"
    prompt += "- For string parameters, "
    prompt += "return only the string value without extra text.\n"
    prompt += "- For boolean parameters, return only true or false.\n"
    prompt += "- If the parameter is named regex, "
    prompt += "return a valid regex pattern.\n"
    prompt += "- For replacing numbers or digits, return regex \\d+.\n"
    prompt += "- In final JSON this regex must appear as \"\\\\d+\".\n"
    prompt += "- For replacing vowels, return regex ([aeiouAEIOU]).\n"
    prompt += "- If the replacement is described as asterisks, return *.\n"
    prompt += "- Do not return words like numbers, "
    prompt += "digits, vowels, or asterisks "
    prompt += "when a regex or symbol is required.\n"
    prompt += "\nExamples:\n"
    prompt += 'User: Replace all numbers in "Hello 34 I am 233" with NUMBERS\n'
    prompt += (
        'Output: {"prompt": "Replace all numbers in \\"Hello 34 I am 233\\" '
        'with NUMBERS", "name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "Hello 34 I am 233", '
        '"regex": "\\d+", "replacement": "NUMBERS"}}\n'
    )
    prompt += "User: Replace all vowels in "
    prompt += "'Programming is fun' with asterisks\n"
    prompt += (
        'Output: {"prompt": "Replace all vowels in \'Programming is fun\' '
        'with asterisks", "name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "Programming is fun", '
        '"regex": "([aeiouAEIOU])", "replacement": "*"}}\n'
    )
    prompt += 'User: "What is the sum of 2 and 3"\n'
    prompt += (
        'Output: {"prompt": "What is the sum of 2 and 3", '
        '"name": "fn_add_numbers", '
        '"parameters": {"a": 2.0, "b": 3.0}}\n'
    )

    prompt += "\nUser: "
    prompt += base_prompt
    prompt += "\nOutput: "
    prompt += json_part
    return prompt


def get_function_from_name(
    name: str,
    functions_list: list[Function],
) -> Function:
    """Return function matching name, or first function."""
    for function in functions_list:
        if name == function.name:
            return function
    return functions_list[0]


def full_fn_list(functions: list[Function]) -> str:
    """Return all functions as prompt text."""
    full_str = ""
    for function in functions:
        full_str += function.fn_to_prompt()
        full_str += "\n"
    return full_str


def all_functions_names(functions: list[Function]) -> list[str]:
    """Return all function names."""
    all_names = []
    for function in functions:
        all_names.append(function.name)
    return all_names


def strip_value(val: Any) -> str:
    """Clean generated value."""
    value = str(val)
    value = value.split("\n")[0]
    value = value.strip('"')
    value = value.strip("'")
    value = value.strip()
    value = value.strip("}")
    value = value.strip(",")
    return value.strip()


def convert_value(val: Any, param_type: str) -> Any:
    """Convert value to expected parameter type."""
    value = strip_value(val)

    if param_type == "number":
        return float(value)

    if param_type == "boolean":
        return value.lower() == "true"

    return value


def is_float(generated_text: str) -> bool:
    """Return True if text is a valid number."""
    value = strip_value(generated_text)

    if value in ["", "-", ".", "-."]:
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False


def argument_is_finished(generated_text: str, param_type: str) -> bool:
    """Check whether argument generation can stop."""
    clean_text = generated_text.strip()

    if param_type == "number":
        return is_float(clean_text)

    if param_type == "boolean":
        return clean_text in ["true", "false"]

    return any(char in clean_text for char in ['}', '"'])


def get_max_logit_index(logits: Any) -> int:
    """Return index of the largest logit."""
    max_logit = float(-inf)
    max_idx = -1

    for idx, logit in enumerate(logits):
        if logit > max_logit:
            max_logit = logit
            max_idx = idx

    return max_idx


def numbers_from_prompt(prompt: str) -> list[str]:
    """Extract numbers from prompt."""
    return re.findall(r"-?\d+(?:\.\d+)?", prompt)


def remove_used_numbers(
    number_candidates: list[str],
    used_numbers: list[float],
) -> list[str]:
    """Remove numbers already used."""
    remaining = []

    for candidate in number_candidates:
        try:
            candidate_float = float(candidate)
        except ValueError:
            remaining.append(candidate)
            continue

        if candidate_float not in used_numbers:
            remaining.append(candidate)

    return remaining


def boolean_candidates() -> list[str]:
    """Return boolean candidates."""
    return ["true", "false"]


def get_candidate_best_token(
    llm: Small_LLM_Model,
    logits: Any,
    generated_text: str,
    candidates: list[str],
) -> int:
    """Return best token from allowed candidates."""
    masked_logits = len(logits) * [-inf]
    active_candidates = []

    for candidate in candidates:
        if candidate.startswith(generated_text):
            active_candidates.append(candidate)

    if not active_candidates:
        active_candidates = candidates

    for candidate in active_candidates:
        rest = candidate[len(generated_text):]

        if rest == "":
            rest = candidate

        token_ids = llm.encode(rest).tolist()[0]

        if token_ids:
            token_id = token_ids[0]
            masked_logits[token_id] = logits[token_id]

    best_idx = get_max_logit_index(masked_logits)

    if best_idx == -1 and active_candidates:
        token_ids = llm.encode(active_candidates[0]).tolist()[0]
        return int(token_ids[0])

    return best_idx


def generate_value(
    llm: Small_LLM_Model,
    full_prompt_parameters: str,
    param_type: str,
    base_prompt: str,
    used_numbers: list[float],
) -> Any:
    """Generate one parameter value."""
    tokens = llm.encode(full_prompt_parameters).tolist()[0]

    max_tokens = 25
    generated_tokens: list[int] = []
    generated_text = ""

    number_candidates = remove_used_numbers(
        numbers_from_prompt(base_prompt),
        used_numbers,
    )

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(tokens + generated_tokens)

        if param_type == "number" and number_candidates:
            clean_generated = strip_value(generated_text)
            max_idx = get_candidate_best_token(
                llm,
                logits,
                clean_generated,
                number_candidates,
            )
        elif param_type == "boolean":
            clean_generated = strip_value(generated_text)
            max_idx = get_candidate_best_token(
                llm,
                logits,
                clean_generated,
                boolean_candidates(),
            )
        else:
            max_idx = get_max_logit_index(logits)

        generated_tokens.append(max_idx)
        generated_text = str(llm.decode(generated_tokens))
        clean_text = strip_value(generated_text)

        if param_type == "number" and number_candidates:
            if clean_text in number_candidates:
                break
        elif param_type == "boolean":
            if clean_text in boolean_candidates():
                break
        elif argument_is_finished(generated_text, param_type):
            break

    if param_type == "string":
        return generated_text.split('"')[0]

    return generated_text


def generate_function(
    llm: Small_LLM_Model,
    prompt: str,
    functions_list: list[Function],
) -> Function:
    """Generate function name with constrained decoding."""
    full_prompt = big_prompt(prompt, functions_list)
    tokens = llm.encode(full_prompt).tolist()[0]

    generated_tokens: list[int] = []
    generated_name = ""
    allowed_names = all_functions_names(functions_list)

    for i in range(10):
        logits = llm.get_logits_from_input_ids(tokens + generated_tokens)
        masked_logits = len(logits) * [-inf]

        allowed_tokens = [
            llm.encode(name).tolist()[0]
            for name in allowed_names
        ]

        for tokens_list in allowed_tokens:
            if len(tokens_list) > i:
                masked_logits[tokens_list[i]] = logits[tokens_list[i]]

        max_idx = get_max_logit_index(masked_logits)

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

    return get_function_from_name(generated_name, functions_list)


def create_functions_list(
    load_functions: list[dict[str, Any]],
) -> list[Function]:
    """Create Function objects."""
    functions_list = []

    for idx, data in enumerate(load_functions):
        function = Function.create_from_dict(data, idx)
        functions_list.append(function)

    return functions_list


def fill_function_parameters(
    llm: Small_LLM_Model,
    selected_function: Function,
    prompt: str,
) -> None:
    """Generate parameters for selected function."""
    used_numbers: list[float] = []

    for index, key in enumerate(selected_function.parameters):
        json_part = selected_function.to_json_parts(prompt, index)
        full_prompt_parameters = prompt_for_param(
            prompt,
            json_part,
            selected_function,
        )

        param_type = selected_function.parameters_type.get(key, "string")
        generated_value = generate_value(
            llm,
            full_prompt_parameters,
            param_type,
            prompt,
            used_numbers,
        )

        converted_value = convert_value(generated_value, param_type)

        selected_function.parameters[key] = converted_value

        if param_type == "number":
            used_numbers.append(float(converted_value))


def process_calls(
    llm: Small_LLM_Model,
    load_calls: list[dict[str, Any]],
    functions_list: list[Function],
) -> list[dict[str, Any]]:
    """Process all prompts."""
    results = []

    for call in load_calls:
        prompt = str(call.get("prompt", ""))
        selected_function = generate_function(llm, prompt, functions_list)

        fill_function_parameters(llm, selected_function, prompt)

        results.append(selected_function.to_dict(prompt))

    return results


def main() -> None:
    """Run pipeline."""
    args = parse_arguments()

    try:
        llm = Small_LLM_Model()

        load_functions = load_json_file(args.functions_definition)
        load_calls = load_json_file(args.input)

        functions_list = create_functions_list(load_functions)
        results = process_calls(llm, load_calls, functions_list)

        save_results(args.output, results)

    except FileNotFoundError as exc:
        print(f"File not found: {exc}")
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON file: {exc}")
    except ValueError as exc:
        print(f"Invalid generated value: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
