NLEM Drug Classification System
AI-Powered Drug Scheduling Assistant for Indian Pharma Industry

Overview

The NLEM Drug Classification System is an AI-powered Q&A application that enables pharmaceutical companies to instantly classify drugs under India's National List of Essential Medicines (NLEM) regulations. Built using Retrieval-Augmented Generation (RAG), the system accepts natural language queries and returns accurate Schedule / Non-Schedule classification along with pricing details.


Business Problem: Major pharmaceutical companies struggle to classify drugs in compliance with NLEM regulations due to the sheer volume of data (88,000+ SKUs), leading to delayed decision-making and regulatory risk.

Features

Natural Language Queries — Ask "Is Paracetamol 500mg tablet Scheduled?" in plain English
NLEM 2022 Classification — Instantly returns Schedule or Non-Scheduled status
Pricing Details — Retrieves MRP, PTR, PTS, and Ceiling Prices from source data
Source Traceability — Every answer includes source document references for auditability
Composition-Aware — Handles complex multi-drug compositions (e.g., HYDROQUINONE + TRETINOIN + HYDROCORTISONE)
Streamlit UI — Clean, browser-based interface with no technical skills required
Offline Vector Search — FAISS-based local vector store for fast and private retrieval

Project Flow

CSV Data → Text Splitting → HuggingFace Embeddings
→ FAISS Vector Store → Retriever → Google PaLM
→ RetrievalQA Chain → Streamlit App

Tech Stack

* Python 3.10+
* Google PaLM
* LangChain
* HuggingFace all-MiniLM-L6-v2
* FAISS
* Streamlit


Use Cases

Classify any drug as Scheduled or Non-Scheduled instantly
Check regulated ceiling prices for NLEM drugs
Assist medical reps and compliance teams with quick lookups
Reduce manual effort in regulatory reporting
