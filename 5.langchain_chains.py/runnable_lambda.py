from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

load_dotenv()

def word_count(text):
    return len(text.split())

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({'topic': 'AI'})

final_result = """{} 
word count - {}""".format(
    result['joke'],
    result['word_count']
)

print(final_result)