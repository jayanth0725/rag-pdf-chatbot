# PDF Chatbot: Conversational RAG Pipeline

Chat with your PDFs using an open-source 7B Large Language Model. Built with LangChain, ChromaDB, and Hugging Face.

This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions about a specific PDF document and receive accurate, context-aware answers with full conversational memory.

**[Try the Live Web App Here](https://rag-pdf-chatbot-jayanth0725.streamlit.app/)**

---

## The Tech Stack

* **User Interface:** Streamlit
* **Language Model:** Qwen/Qwen2.5-7B-Instruct (via Hugging Face API for the web app) and Zephyr-7B-beta (using 4-bit local quantization for the Kaggle notebook)
* **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
* **Vector Database:** ChromaDB
* **Orchestration:** LangChain Classic (LCEL Architecture)

---

## Key Features

* **Conversational Memory:** Utilizes a History-Aware Retriever to understand follow-up questions by rephrasing them into standalone queries based on the chat history.
* **Semantic Document Ingestion:** Uses LangChain's `PyPDFLoader` and `RecursiveCharacterTextSplitter` to extract and split text into overlapping 500-character chunks, preserving context.
* **Serverless Inference:** The web interface connects to the Hugging Face Inference API (`HuggingFaceEndpoint`), allowing it to run smoothly on standard hardware without local GPU requirements.

---

## How to Run Locally

If you want to run this interface locally on your own machine:

1. Clone the repository and navigate to the project directory:
   ```bash
   cd path/to/your/pdf-chatbot-folder
   ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the Streamlit local web server:
    ```bash
    streamlit run app.py
    ```

Start Chatting: Use the sidebar to enter your free Hugging Face API Key and upload your target document.

---

### Getting a Free API Token

To interact with the Zephyr-7B model, you will need a free access token from Hugging Face:
1. Go to your [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
2. Click **Create new token**.
3. Select **Fine-grained** as the token type, give it a name, and click the **Inference** preset button (this grants your app permission to make text-generation calls).
4. Click **Create token**, copy it, and paste it directly into the sidebar of the application!

---

## Development (Kaggle GPU Notebook)

While the Streamlit app relies on cloud APIs for inference, this repository also includes the complete GPU Blueprint as a Jupyter Notebook (pdf-chatbot.ipynb).  This notebook proves the underlying machine learning engineering and serves as a 100% self-hosted, offline testing lab:

1. Hardware: Configured to run on Kaggle Notebooks using P100 or T4x2 GPUs. 

2. Quantization: Demonstrates how to load the massive 7B parameter model directly into VRAM using BitsAndBytesConfig (4-bit quantization). 

3. Local Inference: Utilizes PyTorch and Hugging Face pipeline for direct, hardware-level model execution without relying on external API endpoints.

To experiment with the raw pipeline:

1. Open the Kaggle Notebook.

2. Click Copy & Edit and ensure the Accelerator is set to a GPU tier.

3. Run all cells to process the PDF and execute retrieval chains natively. 