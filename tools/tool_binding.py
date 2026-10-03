from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from rich import print


# Creating llm
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)


# Step 1: Creating a tool
@tool
def get_text_len(text: str) -> int:
    """ Create a tool to return the number of character of the given text"""
    return len(text)


# Step 2: tool binding with llm
llm = get_llm()
llm_with_tool = llm.bind_tools([get_text_len])


def main():

    result1 = llm.invoke("Hello")
    print(result1)

    print("\n\n\n")

    result2 = llm_with_tool.invoke("Hello")
    print(result2)

if __name__ == '__main__':
    main()
