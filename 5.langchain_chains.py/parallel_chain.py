from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()


# We can use any models here.
# Using the same model 3 times is also completely valid.
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm3 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)


model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()


# Two chains run in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})


# The outputs of the parallel chains go into this chain
merge_chain = prompt3 | model3 | parser


# Parallel chain + sequential merge chain
chain = parallel_chain | merge_chain


text = """
Support vector machines (SVMs) are a set of supervised learning methods
used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than
the number of samples.

Uses a subset of training points in the decision function
(called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the
decision function.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples,
avoid over-fitting in choosing Kernel functions and regularization
term is crucial.

SVMs do not directly provide probability estimates.
"""


result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()