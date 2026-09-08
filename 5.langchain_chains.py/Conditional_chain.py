from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# We can use any models here.
# Using the same model for all three is also completely valid.

llm1 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation"
)

llm3 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)

parser = StrOutputParser()


class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model1 | parser2


prompt2 = PromptTemplate(
    template='Reply to this positive feedback with ONE short, natural response. Do not give examples or explanations. Only output the response.\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Reply to this negative feedback with ONE short, polite response. Do not give examples or explanations. Only output the response.\n{feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model2 | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model3 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifier_chain | branch_chain


print(chain.invoke({
    'feedback': 'This is the worse phone i ever bought'
}))

chain.get_graph().print_ascii()