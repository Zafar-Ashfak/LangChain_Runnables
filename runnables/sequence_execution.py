from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

# Step 1: Create the prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in easy words."
)


# Step 2: Initialize the chat model
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)


# Step 3: Create the output parser
parser = StrOutputParser()


def main():
    llm = get_llm()

    print("What's in your mind!")
    query = input("You: ")

    # Step 4: Create the LCEL (LangChain Expression Language) chain
    formatted_prompt = prompt.invoke({
        "topic": query
    })

    response = llm.invoke(formatted_prompt)

    final_output = parser.parse(response.content)

    print(f"AI: {final_output}")


main()
