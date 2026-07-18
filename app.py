import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("Chat with your PDF")

with st.sidebar:
    hf_api_key = st.text_input("Enter Hugging Face API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("Get a free API key from your [Hugging Face Settings](https://huggingface.co/settings/tokens).")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and hf_api_key:
    @st.cache_resource
    def build_rag_pipeline(file_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(file_bytes)
            temp_path = temp.name

        docs = PyPDFLoader(temp_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(chunks, embeddings)

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            huggingfacehub_api_token=hf_api_key,
            temperature=0.1,
            task="text-generation"
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        condense_template = """<|im_start|>system
Given the following chat history and a follow-up question, rephrase the follow-up question to be a standalone question.
Chat History:
{chat_history}<|im_end|>
<|im_start|>user
Follow-up question: {input}<|im_end|>
<|im_start|>assistant
Standalone question:"""
        condense_prompt = PromptTemplate.from_template(condense_template)
        history_aware_retriever = create_history_aware_retriever(llm, retriever, condense_prompt)

        qa_template = """<|im_start|>system
You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Context: {context}<|im_end|>
<|im_start|>user
Chat History:
{chat_history}

Question: {input}<|im_end|>
<|im_start|>assistant
"""
        qa_prompt = PromptTemplate.from_template(qa_template)
        combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    qa_chain = build_rag_pipeline(uploaded_file.getvalue())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                formatted_history = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    formatted_history += f"{role}: {msg['content']}\n"

                response = qa_chain.invoke({"input": prompt, "chat_history": formatted_history})

                answer = response["answer"]

                if "<|im_start|>assistant" in answer:
                    answer = answer.split("<|im_start|>assistant")[-1].strip()
                elif "assistant\n" in answer:
                    answer = answer.split("assistant\n")[-1].strip()

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please enter your Hugging Face API key and upload a PDF to begin.")
