# Forward Bias

**Forward Bias** is a small beginner-friendly **RAG (Retrieval-Augmented Generation)** project written in Python.

It shows the important RAG steps directly instead of hiding them behind a large framework.

## What this project does

The pipeline is:

```text
Your Documents
      |
      v
 Read Text
      |
      v
 Split into Chunks
      |
      v
 Create Embeddings
      |
      v
 Store in FAISS
      |
      v
 User Question
      |
      v
 Similarity Search
      |
      v
 Retrieve Relevant Chunks
      |
      v
 Send Context + Question to Ollama
      |
      v
 Final Answer
```

Supported document types:

- `.txt`
- `.md`
- `.pdf`

## Why RAG?

A normal language model answers mainly from what it learned during training.

RAG adds an external knowledge source.

Instead of asking only:

```text
Question -> LLM -> Answer
```

we use:

```text
Question
   |
   v
Vector Search
   |
   v
Relevant Document Chunks
   |
   v
Question + Context
   |
   v
LLM
   |
   v
Answer
```

This helps the model answer questions about your own documents.

## Project Structure

```text
forward-bias/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── example.txt
└── tests/
    └── test_rag.py
```

## Requirements

Recommended:

- Python 3.10+
- 8 GB RAM or more
- Ollama if you want local LLM answering
- No GPU is required for the basic example

A GPU can speed up local models.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/forward-bias.git
cd forward-bias
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Step 1: Add your documents

Put your files inside:

```text
data/
```

For example:

```text
data/
├── company_policy.pdf
├── notes.txt
└── handbook.md
```

## Step 2: Create the vector database

Run:

```bash
python app.py index --source data
```

The program will:

1. Read the documents
2. Extract their text
3. Split the text into chunks
4. Convert chunks to embeddings
5. Normalize the embeddings
6. Store them in a FAISS vector index
7. Save metadata for each chunk

Generated files are stored in:

```text
storage/
├── index.faiss
├── chunks.json
└── config.json
```

The `storage/` directory is ignored by Git because it can always be regenerated.

## Step 3: Test retrieval without an LLM

You can first see whether retrieval works:

```bash
python app.py ask "What is RAG?" --no-llm
```

This prints the chunks that FAISS believes are most relevant.

This is useful because it lets you debug the **retrieval** part separately from the LLM.

## Step 4: Use a local LLM with Ollama

Install Ollama from its official website.

Then pull a model:

```bash
ollama pull llama3.2:3b
```

Ask a question:

```bash
python app.py ask "What is RAG?"
```

The program retrieves relevant chunks and sends them with your question to Ollama.

## Example

The included file:

```text
data/example.txt
```

contains a small explanation of RAG.

First index it:

```bash
python app.py index --source data
```

Then ask:

```bash
python app.py ask "Why do we use embeddings?"
```

You should receive an answer based on the example document.

## Important RAG Concepts Used Here

### 1. Document Loading

The project reads text from TXT, Markdown, and PDF files.

### 2. Chunking

Large documents are broken into smaller pieces.

Default values:

```text
chunk size = 800 characters
overlap    = 120 characters
```

Overlap helps preserve information that appears near chunk boundaries.

### 3. Embeddings

This project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

An embedding is a vector representation of text.

Texts with similar meaning tend to have embeddings that are closer together.

### 4. Vector Database

This project uses:

```text
FAISS
```

FAISS performs fast similarity searches over vectors.

### 5. Retrieval

When you ask a question:

```text
What is retrieval?
```

the question is also converted to an embedding.

FAISS compares that vector with the vectors created from your document chunks.

The most similar chunks are returned.

### 6. Augmentation

The retrieved text is added to the prompt.

The LLM receives something similar to:

```text
CONTEXT:
<retrieved text>

QUESTION:
<user question>

ANSWER:
```

### 7. Generation

Ollama sends the final prompt to a local LLM.

The model produces the answer using the retrieved context.

## Why this project does not use LangChain

This repository intentionally avoids LangChain so that you can see the core RAG pipeline.

Once you understand this version, you can later create another version using:

- LangChain
- LlamaIndex
- ChromaDB
- Qdrant
- Pinecone
- Weaviate

## Run Tests

```bash
pytest
```

The tests check the basic chunking and prompt-building behavior.

## Example Commands

Index all documents:

```bash
python app.py index --source data
```

Search only:

```bash
python app.py ask "Explain vector search" --no-llm
```

Retrieve more chunks:

```bash
python app.py ask "Explain vector search" --top-k 6 --no-llm
```

Use a different Ollama model:

```bash
python app.py ask "Explain RAG" --model qwen2.5:7b
```

## What You Learn From This Repository

After completing this project, you should understand:

- What RAG is
- Why documents are chunked
- What embeddings are
- What a vector database does
- What semantic search means
- How retrieval works
- How retrieved context is added to a prompt
- How an LLM generates a grounded answer
- How FAISS works in a basic local RAG application

## Next Improvements

Good next steps are:

1. Add a Streamlit web interface
2. Add source citations to answers
3. Add conversation memory
4. Experiment with different chunk sizes
5. Compare embedding models
6. Add metadata filtering
7. Use ChromaDB or Qdrant
8. Add reranking
9. Add hybrid keyword + vector search
10. Measure retrieval accuracy
11. Add prompt-injection defenses
12. Add a RAG evaluation dataset

## License

MIT License.
