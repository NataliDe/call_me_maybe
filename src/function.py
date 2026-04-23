import json
class Function:
    def __init__(self,idx, name, prompt, description, parameters, parameters_type, returns):
        self.idx = idx
        self.name = name
        self.prompt = prompt
        self.description = description
        self.parameters = parameters or {} #beda typ
        self.parameters_type = parameters_type or {}
        self.returns = returns

    def fn_to_prompt(self):
        return (f"{self.name} - {self.description} Parameters: {str(self.parameters)}")
    
    def return_names(self):
        return self.name

    def to_json(self):
        return json.dumps(
        {
            "prompt": self.prompt,
            "name": self.name,
            "parameters": self.parameters
        })
    


    @staticmethod
    def get_params(data):
        types_dict = {}
        parameters = data.get("parameters", {})
        for parameter in parameters:
            types_dict[parameter] = parameters.get(parameter).get("type")
        return types_dict

    @staticmethod
    def params_to_none(data):
        types_dict = {}
        parameters = data.get("parameters", {})
        for parameter in parameters:
            types_dict[parameter] = None
        return types_dict


    @classmethod
    def create_from_dict(cls, data, idx):
        return cls(
            idx=idx,
            name=data.get("name"),
            description=data.get("description"),
            prompt=data.get("prompt"),
            parameters=cls.params_to_none(data),
            parameters_type = cls.get_params(data),
            returns=data.get("returns")
        )

    