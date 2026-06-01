# 📊 FinSight RAG

### AI-Powered Banking & Economic Research Assistant

FinSight RAG is a Generative AI (GenAI) and Retrieval-Augmented Generation (RAG) research assistant designed for banking, financial, economic, and policy analysis.

By combining Large Language Models (LLMs), vector databases, and retrieval-based reasoning, the application enables users to upload single or multiple documents and transform lengthy PDF reports into actionable insights through summarization, risk identification, report comparison, and source-grounded question answering.

---

## Features

* 📄 Individual Report Summaries
* 📚 Combined Multi-Report Summaries
* ⚖️ Multi-Report Comparison
* 🚨 Risk Identification & Analysis
* 💬 Conversational Question Answering
* 📖 Source-Aware Responses

---

## Technology Stack

**AI & Retrieval**

* OpenAI GPT-4o-mini
* OpenAI Embeddings
* LangChain
* Chroma Vector Database
* MMR (Max Marginal Relevance) Retrieval

**Application**

* Python
* Streamlit

**Document Processing**

* PyPDF
* Recursive Character Text Splitting

---

## Architecture

PDF Reports

↓

Document Chunking

↓

OpenAI Embeddings

↓

Chroma Vector Database

↓

Retrieval Layer

* MMR Retrieval for Question Answering
* Thematic Retrieval for Summaries
* Risk-Focused Retrieval for Risk Analysis
* Multi-Document Retrieval for Report Comparison

↓

GPT-4o-mini

↓

Grounded Answers & Insights

---

## Project Objective

This project demonstrates practical applications of:

* Retrieval-Augmented Generation (RAG)
* OpenAI Embeddings
* Vector Databases
* MMR-Based Retrieval
* Multi-Document Analysis
* AI-Assisted Economic Research

The goal is to create an AI-powered research assistant capable of transforming complex economic and financial documents into actionable insights for analysts, consultants, bankers, and policy professionals.
