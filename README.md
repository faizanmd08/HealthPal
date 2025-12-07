# HealthPal: Grounded Medical Q&A Chatbot 🩺

> A specialized AI assistant designed to provide accurate, source-grounded answers to clinical queries by synthesizing data from authoritative medical textbook.

## 📖 Overview

HealthPal addresses a critical challenge in modern Generative AI: **hallucination in high-stakes domains**. General-purpose Large Language Models (LLMs) often generate plausible but incorrect medical advice when relying solely on their pre-trained knowledge.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that "grounds" the AI. Instead of inventing answers, HealthPal retrieves specific excerpts from a curated medical knowledge base (e.g., _The Gale Encyclopedia of Medicine_) and synthesizes a response based _only_ on that verified information. It is designed to act as a trustworthy digital assistant for students and medical professionals.

## ✨ Key Features

- **Trustworthy & Grounded:** The chatbot is engineered to answer questions strictly using the provided medical context. If the answer isn't in the source material, it will explicitly state that it doesn't know, rather than guessing.
- **Semantic Search:** Unlike keyword-based search (which fails if you don't use the exact right words), HealthPal uses **vector embeddings** to understand the intent and meaning behind a user's question (e.g., matching "high temperature" to "fever").
- **Source Transparency:** Every response is generated based on retrieved text chunks, allowing for clear traceability back to the original medical documentation.
- **Context-Aware:** Capable of understanding complex medical terminology and synthesizing information from multiple sections of a textbook to provide a cohesive answer.

## ⚙️ How It Works (The Architecture)

The system operates on a sophisticated **RAG (Retrieval-Augmented Generation)** architecture, split into two main workflows:

### 1. The Knowledge Ingestion Pipeline

Before the chatbot can answer questions, it builds its "brain" from raw data:

- **Extraction:** Medical PDF textbooks are processed to extract raw text while preserving logical flow.
- **Semantic Chunking:** The text is split into smaller, meaningful segments (chunks). Special care is taken to keep related concepts (like symptoms and treatments) together.
- **Vector Embedding:** Each chunk is converted into a high-dimensional mathematical vector (a list of numbers) that represents its semantic meaning.
- **Indexing:** These vectors are stored in a specialized **Vector Database** (Pinecone), creating a searchable index of medical knowledge.

### 2. The Retrieval & Generation Pipeline

When a user asks a question, the system performs the following in real-time:

- **Query Embedding:** The user's question is converted into the same vector format.
- **Similarity Search:** The database finds the text chunks that are mathematically closest (most similar in meaning) to the user's question.
- **Synthesis:** The retrieved medical chunks + the user's question + strict instructions are sent to the **LLM (Large Language Model)**.
- **Response:** The LLM acts as a summarizer and synthesizer, crafting a natural language answer based _only_ on the retrieved facts.

## 🛠️ Technology Stack

This project leverages a modern AI stack designed for speed, accuracy, and scalability:

- **Large Language Model (LLM):** **Google Gemini 2.0 Flash** – Selected for its high speed, reasoning capabilities, and large context window.
- **Vector Database:** **Pinecone (Serverless)** – A managed vector database used to store and retrieve millions of high-dimensional text embeddings instantly.
- **Orchestration Framework:** **LangChain** – The glue code that manages the flow between the user, the database, and the LLM.
- **Embedding Model:** **Sentence Transformers (`all-MiniLM-L6-v2`)** – An open-source model that converts text into dense vectors for semantic comparison.
- **Backend Framework:** **Flask (Python)** – A lightweight web server that handles API requests and serves the chatbot interface.
- **Data Processing:** **PyMuPDF** – For robust extraction of text and metadata from PDF documents.

## 🔮 Future Roadmap

- **Multimodal Capabilities:** Integrating image recognition to allow users to upload medical diagrams or skin condition photos for analysis.
- **Voice Interface:** Adding Speech-to-Text (STT) and Text-to-Speech (TTS) to allow hands-free voice queries for medical practitioners.
- **Citation Links:** Enhancing the UI to display the exact page number and paragraph source for every claim made by the bot.
