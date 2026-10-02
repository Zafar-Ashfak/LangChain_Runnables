from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

tavily_tool = TavilySearch(
    max_results=5
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
            """
                You are an expert news summarizer AI assistant.
                Summarize the news in bullet points in simple english words.
            """
     ),
    ("human", "{news}")
])

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

tavily_news = tavily_tool.run(tool_input="CJP protest on 2 Oct in Mumbai.")

def main():
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "news": tavily_news
    })

    print(response)

if __name__ == '__main__':
    main()



