# RAG PDF Chatbot: Chat with your Documents

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions about a specific PDF document and receive accurate, context-aware answers. 

This project runs entirely on a free Kaggle GPU using open-source models, requiring no paid API keys.

## The Tech Stack

* **Language Model:** Zephyr-7B-beta (Quantized to 4-bit)
* **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
* **Vector Database:** ChromaDB
* **Orchestration:** LangChain
* **Hardware:** Kaggle Notebook (P100/T4 GPU x2)

## How It Works

1. **Document Ingestion:** Uses LangChain's `PyPDFLoader` to extract text from `Retrieval-augmented_generation.pdf]` (downloaded from Wikipedia).
2. **Chunking:** Splits the text into overlapping 500-character chunks to bypass LLM memory limits while preserving context.
3. **Vectorization:** Converts text chunks into mathematical embeddings and stores them in a local Chroma vector database.
4. **Retrieval & Generation:** When a user asks a question, the system retrieves the top 3 most relevant text chunks and passes them to the Zephyr-7B model to synthesize a final answer.

## Results & Example

**Document Used:** [A Wikipedia page about "Retrieval-augmented_generation" in PDF format]

**Question:** 
> "[Summarize the core concepts and operational process of Retrieval-Augmented Generation based on the text.]"

**Generated Answer:** 
"Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by allowing them to access and utilize external data beyond their pre-existing training data. The process involves the following core concepts and operational steps:

1. External documents: LLMs refer to a specified set of external documents to supplement information from their training data.

2. User query: LLMs respond to user queries based on the combined information from the external documents and their training data.

3. Information retrieval: LLMs retrieve relevant information from the external documents based on the user query.

4. Prompt: LLMs are prompted with a combination of the user query and the retrieved information to generate tailored output.

5. Output: LLMs generate output based on both the user query and the retrieved information.

6. Improvements: Some models incorporate additional steps to improve output, such as re-ranking of retrieved information, context selection, and fine-tuning.

RAG is used in applications where generated responses need to be grounded in external or frequently updated information, such as healthcare."

## How to Run It Yourself

Because this relies on a 7-billion parameter language model, it requires a GPU to run efficiently. The easiest way to test this code is via Kaggle:

1. Click [here](https://www.kaggle.com/code/jayanthraveendra/pdf-chatbot) to open the Kaggle Notebook.
2. Click **Copy & Edit** in the top right corner.
3. Ensure the Accelerator is set to **GPU P100** or **GPU T4x2** in the session options.
4. Upload the PDF in /data to the Kaggle environment or upload your own PDF to the environment and change the file path in Cell 4. You can change the question as well in Cell 6.
5. Run all cells!
