from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

short_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in 3 to 4 lines."
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in detail"
)


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
        "short": short_prompt | llm | parser,
        "detailed": detailed_prompt | llm | parser
    })

    response = chain.invoke({
        "topic": "What is Machine Learning?"
    })

    print(f"{'-' * 30} short Answer {'-' * 30}\n {response['short']}\n")
    print(f"{'-' * 30} Detailed Answer {'-' * 30}\n {response['detailed']}")

if __name__ == '__main__':
    main()