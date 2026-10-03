from langchain.tools import tool

@tool
def greet(name : str) -> str:
    """This tool helps to greet user"""

    return f"Hello, {name}.\nWelcome to AI world!"

result = greet.invoke("Sadie Sink")
print(result)