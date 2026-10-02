
# Code Generator and Explanation AI Assistant

# This program uses LangChain Expression Language (LCEL) to:
# 1. Generate code based on a user-provided topic.
# 2. Explain the generated code in simple words.
# 3. Use RunnableParallel and RunnablePassthrough to return
#    both the generated code and its explanation.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart code generator. Generate code for: {topic}"),
    ("human", "{topic}")
])

explained_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert code explainer. Explain the code in simple words."),
    ("human", "{code}")
])

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

def main():
    print("\nCode generator and explanation AI Assistant.")
    topic = input("Enter a topic: ")
    llm = get_llm()

    chain1 = code_prompt | llm | StrOutputParser()

    chain2 = RunnableParallel({
        "code": RunnablePassthrough(),
        "explanation": explained_prompt | llm | StrOutputParser()
    })

    final_chain = chain1 | chain2

    response = final_chain.invoke({
        "topic": topic
    })

    print("\nGenerated Code:\n")
    print(response["code"])

    print("\nCode Explanation:\n")
    print(response["explanation"])

if __name__ == '__main__':
    main()

