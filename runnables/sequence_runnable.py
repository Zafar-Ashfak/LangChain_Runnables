from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

# Prompt template
prompt = PromptTemplate.from_template("Explain the {topic} in easy words.")

# Chat Model
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

# Output parser
parser = StrOutputParser()

def main():
    llm = get_llm()
    print("What's in your mind!")
    query = input("You: ")

    chain = prompt | llm | parser
    response = chain.invoke(query)
    print(f"AI: {response}")

main()