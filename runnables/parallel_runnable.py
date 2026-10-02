from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

short_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
            You are a good AI Assistant.
            Explain the topic in 3 to 4 sentences
        """
     ),
    ("human", "{topic}")
])

detailed_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
            You are an Expert AI Assistant.
            Explain the {topic} in easy words and in detail.
        """
     ),
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
    llm = get_llm()

    chain = RunnableParallel({
        "short" : RunnableLambda(lambda x : x ['short']) | short_prompt | llm | parser,
        "detailed" : RunnableLambda(lambda x : x ['detailed']) | detailed_prompt | llm | parser
    })

    response = chain.invoke({
        "short": {
            "topic": "What is computer vision?"
        },

        "detailed": {
            "topic": "What is NLP (Natural Language Processing?)"
        }
    })

    print(f"\n{'-' * 40} Short Response {'-' * 40}\n {response['short']}\n\n")
    print(f"{'-' * 40} Detailed Response {'-' * 40}\n {response['detailed']}")

if __name__ == '__main__':
    main()

