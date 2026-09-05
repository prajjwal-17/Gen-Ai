from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_Template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human', 'Explain me in simple terms, what is {topic}')
])

prompt = chat_Template.invoke({'domain':'cricket' , 'topic':'Doosra'})

print(prompt)