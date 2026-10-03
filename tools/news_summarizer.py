from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer AI assistant. Summarize the {news} in easy and simple words in bullet points."),
    ("human", "{news}")
])


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

tavily = TavilySearch(max_results=5)

def main():
    news = "What is latest news of CJP protest in Mumbai and Jantar Mantar Delhi on 3 Oct 2026"
    tavily_news = tavily.run(tool_input=news)

    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "news": tavily_news
    })

    print(result)

if __name__ == '__main__':
    main()


