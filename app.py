import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


st.set_page_config(page_title="Medical Assistant", page_icon="🩺", layout="wide")


st.markdown("""
<style>
    /* Hide the default Streamlit top menu and footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make the chat text slightly larger and more readable */
    .stChatMessage {font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100) # Cute medical icon
    st.title("About this Bot")
    st.info(
        "This AI Medical Assistant is powered by a custom medical database. "
        "It retrieves information from clinical documents to answer your health-related questions."
    )
    st.warning(
        "**Disclaimer:** This tool is for informational purposes only and does not provide medical advice. "
        "Always consult with a qualified healthcare provider."
    )
    st.divider()
    st.caption("Powered by LangChain, Groq & Streamlit")

# --- Main Page UI ---
st.title("🩺 Your Virtual Medical Assistant")
st.write("Hello! I'm here to help you understand medical conditions, symptoms, and treatments based on our clinical library.")


@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def setup_qa_chain(db):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.6, # Slightly increased for a warmer, more natural tone
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    #
    CUSTOM_PROMPT_TEMPLATE = """
    You are a friendly, empathetic, and highly knowledgeable Virtual Medical Assistant. 
    Use the provided pieces of information to answer the user's question clearly and gently.
    
    Guidelines:
    1. Start with a warm, supportive tone.
    2. Break down complex medical information into easy-to-understand terms.
    3. Use bullet points and bold text to organize your answer so it is easy to read.
    4. If the context does not contain the answer, kindly apologize and state that you cannot provide information outside of your medical documents.
    5. Always end your response with a brief reminder to consult a real doctor for medical advice.

    Context: {context}
    
    Question: {question}

    Assistant's Friendly Answer:
    """
    
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

db = load_vector_db()
qa_chain = setup_qa_chain(db)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! 👋 I am your Virtual Medical Assistant. What would you like to know about today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if user_query := st.chat_input("Ask me about a symptom, disease, or treatment..."):
    
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process the query
    with st.chat_message("assistant"):
        with st.spinner("Let me look into the medical documents for you..."):
            response = qa_chain.invoke({'query': user_query})
            result = response["result"]
            source_docs = response["source_documents"]
            
            
            st.markdown(result)
            
            # Show sources neatly
            with st.expander("📚 View Clinical Sources"):
                for i, doc in enumerate(source_docs):
                    st.info(f"**Document {i+1}** (Page {doc.metadata.get('page', 'Unknown')})")
                    st.write(f"*{doc.page_content}*")


    st.session_state.messages.append({"role": "assistant", "content": result})