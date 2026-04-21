class Function:
    def __init__(self,idx, name, prompt, description, parameters, parameter_values, returns):
        self.idx = idx
        self.name = name
        self.prompt = prompt
        self.description = description
        self.parameters = parameters or {} #beda typy
        #self.parameter_values = parameter_values or {} # zawartosc
        self.returns


    @staticmethod
    def get_params(data):
        types_dict = {}
        parameters = data.get("parameters", {})
        for parameter in parameters:
            types_dict[parameter] = parameters.get(parameter).get("type")
        return types_dict

    @classmethod
    def create_from_dict(cls, data, idx):
        return cls(
            idx=idx,
            name=data.get("name"),
            description=data.get("description"),
            prompt=data.get("prompt"),
            parameters=get_params(data)
            #parameter_values=??
            returns=data.get("returns")
        )

    