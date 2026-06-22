from typing import Any

from openai import OpenAI


class ReactAgent:
    def __init__(self, system="") -> None:
        self.system = system
        self.messages = []
        self.model = "google/gemma-4-e4b"
        self.client: OpenAI = self.initialiseClient()
        self.tools = {}
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        if result[-5:] != "PAUSE":
            return result
        result = result[:-6]
        result = result.split("\n")
        result = result[-1]
        action, tool_name, *args = result.split(":")
        tool_name = tool_name.strip()
        result = f"Observation : Unknown action: {tool_name}: {args}"
        if tool_name in self.tools.keys():
            try:
                observation = self.tools[tool_name](args)
                result = f"Observation: {observation}"
            except Exception as e:
                print(e)
                result = f"Observation : Failed action: {tool_name}: {args}"
                return result
        else:
            result = f"Observation: action not found: {tool_name}: {args}"
            return result
        return self.__call__(result)
    
    def initialiseClient(self):
        client = OpenAI(base_url="http://192.168.1.8:1234/v1", api_key="shubh-local")
        return client
        
    
    def execute(self) -> str:
        completion = self.client.chat.completions.create(model=self.model, messages=self.messages, temperature=0)
        result = completion.choices[0].message.content
        return result if result else ""
    
    def add_tool(self, key, val):
        self.tools[key] = val
    
    def call_tool(self, name, *args):
        print(f"Calling {name}")
        if name not in self.tools.keys():
            return f"Unknown action: {name}: {args}"
        result = self.tools[name](args)
        return f"Observation: {result}"
    



prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

def calculate(what):
    what = what[0].strip()
    print(f"[!] Executing calculate for {what}")
    return eval(what)

def average_dog_weight(name):
    name = name[0].strip()
    print(f"[!] Executing average_dog_weight with {name}")
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")


mybot = ReactAgent(system=prompt)
mybot.add_tool("calculate", calculate)
mybot.add_tool("average_dog_weight", average_dog_weight)
question = """I have 7 dogs, 3 border collie and 4 scottish terrier. \
Unfortunately one border collie died, now What is their combined weight and average weight"""
print(mybot(question))
