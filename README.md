*This project has been created as part of the 42 curriculum by <ndemkiv>.*

# call me maybe

## Description

This project translates natural-language prompts into structured function calls.
For every input prompt, the program writes a JSON object containing the original
prompt, the selected function name, and all required arguments with the types
specified in `functions_definition.json`.

## Instructions

Install dependencies:

```bash
make install
```

Run with default paths:

```bash
make run
```

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/functions_calling_results.json
```

Run checks:

```bash
make lint
```

## Algorithm explanation

The function name is chosen by the LLM, but token selection is constrained.
The program encodes every available function name and, at each decoding step,
allows only tokens that can still complete one of those names. Invalid tokens are
excluded by assigning them no chance of being selected. This keeps the selected
function inside the schema instead of trusting the model to write free-form JSON.

After the function is selected, each required parameter is generated separately.
The final JSON is not copied from the model output. Instead, the program converts
the generated values to the declared parameter types and serializes the result
with `json.dump`, which guarantees a parseable JSON file with only the required
keys.

## Design decisions

- Pydantic models validate function definitions, prompts, and output objects.
- The CLI follows the subject format: `uv run python -m src` with optional input
  and output paths.
- A fresh parameter dictionary is created for every prompt, so results do not leak
  values from previous calls.
- Errors from missing files, invalid JSON, and invalid schemas are caught and
  displayed as clear messages.

## Performance analysis

The name decoder is fast because it only compares the current token prefix with
encoded function names. Parameter generation is greedy and limited by a maximum
token count to avoid long or stuck generations. The output step is deterministic,
so the produced file remains valid JSON even when the model output is imperfect.

## Challenges faced

Small LLMs often produce extra prose or malformed JSON. To reduce this risk, the
implementation constrains the function-name decoding and builds the final JSON
programmatically. Another challenge was avoiding shared mutable state between
prompts, which is solved by creating fresh parameter dictionaries per result.

## Testing strategy

The program should be tested with valid inputs, missing files, invalid JSON,
empty strings, large numbers, booleans, and prompts that require functions with
multiple parameters. The produced file should be parsed again with Python's JSON
module and compared with the parameter types from the definitions file.

## Resources

- Python `json` documentation
- Python `argparse` documentation
- Pydantic documentation
- The project subject PDF
- AI was used to review formatting requirements, improve error handling.
