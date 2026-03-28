import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

load_dotenv()
# Step 1: Setup LLM (Groq)
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        groq_api_key=os.getenv("GROQ_API_KEY")  )


# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Step 3: Load FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)


# Step 5: Query loop
while True:
    user_query = input("\nWrite Query Here (or 'exit'): ")

    if user_query.lower() == "exit":
        break

    response = qa_chain.invoke({'query': user_query})

    print("\nRESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:\n", response["source_documents"])