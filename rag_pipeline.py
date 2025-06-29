from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load the LLM (Groq-backed DeepSeek)
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Perform similarity search over the vector DB
def retrieve_docs(query, db):
    return db.similarity_search(query)

# Combine retrieved chunks into a single context string
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Prompt to ensure answers are grounded in context only
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't make anything up.
Don't provide anything outside the given context.
Question: {question}
Context: {context}
Answer:
"""

# Generate final LLM response
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})
