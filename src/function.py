"""Function model used by the function-calling generator."""

from copy import copy
import json
from typing import Any

from pydantic import BaseModel


class Function(BaseModel):
    """Store one available function and help format partial JSON prompts."""

    idx: int
    name: str
    prompt: str | None = None
    description: str | None = None
    parameters: dict[str, Any]
    parameters_type: dict[str, str]
    returns: dict[str, Any] | None = None

    def fn_to_prompt(self) -> str:
        """Return a short text description of the function for the LLM."""
        return (f"{self.name} - {self.description} Parameters: "
                f"{self.parameters_type}")

    @staticmethod
    def param_part(
        params: dict[str, Any],
        types: dict[str, str],
        idx: int,
    ) -> str:
        """Build the beginning of the JSON parameters object."""
        params_str = ', "parameters": {'
        for index, key in enumerate(params):
            if index <= idx:
                params_str += f'"{key}": '
            if index == idx:
                if types[key] == "string":
                    params_str += '"'
            if index < idx:
                value = params[key]
                params_str += f'"{value}", '
        return params_str

    def to_json_parts(self, prompt: str, idx: int) -> str:
        """Return the fixed JSON prefix before generating one parameter."""
        first_part = str(
            json.dumps({"prompt": prompt, "name": self.name})).strip("}")
        second_part = self.param_part(
            self.parameters, self.parameters_type, idx)
        return first_part + second_part

    def to_dict(self, prompt: str) -> dict[str, Any]:
        """Return the final output dictionary for one prompt."""
        return {
            "prompt": prompt,
            "name": self.name,
            "parameters": copy(self.parameters),
        }

    def return_names(self) -> str:
        """Return the function name."""
        return self.name

    def to_json(self, prompt: str) -> str:
        """Return the final output as JSON text."""
        return json.dumps(
            {
                "prompt": prompt,
                "name": self.name,
                "parameters": self.parameters,
            }
        )

    @staticmethod
    def get_params(data: dict[str, Any]) -> dict[str, str]:
        """Extract parameter names and types from the input definition."""
        types_dict: dict[str, str] = {}
        parameters = data.get("parameters", {})
        for parameter in parameters:
            types_dict[parameter] = parameters.get(
                parameter, {}).get("type", "string")
        return types_dict

    @staticmethod
    def params_to_none(data: dict[str, Any]) -> dict[str, Any]:
        """Create an empty parameter-value dictionary with the same keys."""
        params_dict: dict[str, Any] = {}
        parameters = data.get("parameters", {})
        for parameter in parameters:
            params_dict[parameter] = None
        return params_dict

    @classmethod
    def create_from_dict(cls, data: dict[str, Any], idx: int) -> "Function":
        """Create a Function object from one JSON function definition."""
        return cls(
            idx=idx,
            name=str(data.get("name", "")),
            description=data.get("description"),
            prompt=data.get("prompt"),
            parameters=cls.params_to_none(data),
            parameters_type=cls.get_params(data),
            returns=data.get("returns"),
        )
