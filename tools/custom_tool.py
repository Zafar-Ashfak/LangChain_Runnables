from langchain.tools import tool

@tool
def is_eligible(name: str, age: int):
    """Create a greeting tool for a user"""
    if age < 18:
        return f"Hello, {name}.\nSorry, you are not eligible to vote."
    else:
        return f"Hello, {name}. You are eligible to vote."

def main():
    name = input("Enter your name: ")
    age = input("Enter your age: ")

    result = is_eligible.invoke({
        "name": name,
        "age": age
    })

    print(result)

if __name__ == '__main__':
    main()