from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system",
            """
                You are a helpful AI assistant.
                Explain the topic in easy words.
            """),
    ("human", "{topic}")
])

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

parser = StrOutputParser()

def main():
    print("What's in your mind!")
    topic = input("You: ")

    llm = get_llm()

    chain = prompt | llm | parser

    response = chain.invoke({
        "topic": topic
    })

    print(f"AI: {response}")


if __name__ == '__main__':
    main()