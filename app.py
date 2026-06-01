import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="FinSight RAG", page_icon="📊", layout="wide")

st.title("📊 FinSight RAG")
st.subheader("Banking & Economic Research Assistant")

st.write(
    "Upload banking, financial, policy, or economic reports. "
    "FinSight can summarize individual reports, create a combined summary, compare reports, "
    "identify risks, and answer follow-up questions using history-aware retrieval."
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

uploaded_files = st.file_uploader(
    "Upload PDF reports",
    type=["pdf"],
    accept_multiple_files=True
)

def load_pdfs(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name

        all_docs.extend(docs)

    return all_docs

def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

def build_context(docs):
    return "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    ])

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def rewrite_followup_question(question, chat_history):
    """
    Rewrites vague follow-up questions into standalone questions before retrieval.
    Example:
    Previous question: "What does the IMF 2026 report say about inflation?"
    Follow-up: "Is it worrisome?"
    Rewritten: "Is the inflation outlook in the IMF 2026 report worrisome?"
    """

    if not chat_history:
        return question

    history_text = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in chat_history[-3:]
    ])

    prompt = ChatPromptTemplate.from_template("""
You are a question rewriting assistant for a RAG system.

Rewrite the user's latest question into a standalone question using the chat history.

Rules:
- Keep the meaning unchanged.
- Do not answer the question.
- If the question is already standalone, return it as-is.
- Make vague references like "it", "this", "that", "they", "these risks", "the report", or "the above" explicit.
- Preserve any report year, institution, topic, or entity from the chat history if needed.

Chat History:
{history}

Latest User Question:
{question}

Standalone Question:
""")

    chain = prompt | get_llm()

    response = chain.invoke({
        "history": history_text,
        "question": question
    })

    return response.content.strip()

def answer_question(vector_store, question, chat_history):
    standalone_question = rewrite_followup_question(question, chat_history)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30}
    )

    relevant_docs = retriever.invoke(standalone_question)
    context = build_context(relevant_docs)

    history_text = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in chat_history[-3:]
    ])

    prompt = ChatPromptTemplate.from_template("""
You are FinSight RAG, a banking and economic research assistant.

Use ONLY the retrieved document context to answer the user's question.
Use chat history only to understand follow-up questions.

If the answer is not available in the retrieved context, say:
"I could not find enough information in the uploaded documents."

Important:
- Do not answer from general knowledge.
- Be specific and business-friendly.
- Cite source document and page number when possible.
- If multiple uploaded reports are relevant, mention which source each point comes from.

Return:
1. Direct Answer
2. Key Supporting Points
3. Source References

Original User Question:
{original_question}

Standalone Retrieval Question:
{standalone_question}

Chat History:
{history}

Retrieved Document Context:
{context}
""")

    chain = prompt | get_llm()

    response = chain.invoke({
        "original_question": question,
        "standalone_question": standalone_question,
        "history": history_text,
        "context": context
    })

    return response.content, relevant_docs, standalone_question

def generate_individual_report_summaries(vector_store, file_names):
    summaries = {}

    for file_name in file_names:
        docs = vector_store.similarity_search(
            "overview executive summary outlook growth inflation risks policy recommendations financial conditions",
            k=14,
            filter={"source": file_name}
        )

        context = build_context(docs)

        prompt = ChatPromptTemplate.from_template("""
You are FinSight RAG, a senior economic research assistant.

Create an individual executive summary for this report using ONLY the retrieved excerpts.

Report Name:
{file_name}

Return:
1. Report Overview
2. Main Economic Themes
3. Growth and Inflation Outlook
4. Key Risks
5. Policy Implications
6. Banking and Financial Sector Relevance
7. 5 Bullet Summary for Senior Management
8. Source References

Context:
{context}
""")

        chain = prompt | get_llm()

        response = chain.invoke({
            "file_name": file_name,
            "context": context
        })

        summaries[file_name] = {
            "summary": response.content,
            "docs": docs
        }

    return summaries

def generate_combined_report_summary(vector_store):
    docs = vector_store.similarity_search(
        "overview executive summary global outlook economic growth inflation risks policy recommendations financial conditions",
        k=22
    )

    context = build_context(docs)

    prompt = ChatPromptTemplate.from_template("""
You are FinSight RAG, a senior economic research assistant.

Create a combined executive summary using ONLY the retrieved excerpts from the uploaded reports.

Important:
- This is a combined summary across uploaded documents.
- If the retrieved context includes multiple reports, distinguish the source documents where possible.
- If one report appears more represented than another, mention that the summary is based on the retrieved excerpts available.
- Do not use general knowledge.

Return:
1. Combined Report Overview
2. Main Economic Themes Across Uploaded Reports
3. Growth and Inflation Outlook
4. Key Risks
5. Policy Implications
6. Banking and Financial Sector Relevance
7. 5 Bullet Summary for Senior Management
8. Source References

Context:
{context}
""")

    chain = prompt | get_llm()
    response = chain.invoke({"context": context})

    return response.content, docs

def identify_key_risks(vector_store):
    docs = vector_store.similarity_search(
        "risks downside risks vulnerabilities uncertainty financial stability inflation growth debt geopolitical trade banking sector",
        k=22
    )

    context = build_context(docs)

    prompt = ChatPromptTemplate.from_template("""
You are FinSight RAG, a banking and macroeconomic risk analyst.

Using ONLY the retrieved document context, identify the major risks discussed in the uploaded reports.

Important:
- If multiple reports are uploaded, mention which report each risk comes from when possible.
- Do not use general knowledge.

Return:
1. Top Risks
2. Why Each Risk Matters
3. Potential Impact on Banks or Financial Institutions
4. Risk Severity: Low / Medium / High
5. Source References

Context:
{context}
""")

    chain = prompt | get_llm()
    response = chain.invoke({"context": context})

    return response.content, docs

def compare_reports(vector_store, comparison_question):
    search_query = (
        comparison_question
        + " compare risks outlook inflation growth policy concerns "
        + "global economy downside risks financial conditions uncertainty"
    )

    docs = vector_store.similarity_search(search_query, k=28)
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_template("""
You are FinSight RAG, a senior macroeconomic research analyst.

Use ONLY the retrieved context below to compare the uploaded reports.

Important:
- The uploaded reports may be from different years or institutions.
- Identify which points relate to which report/year based on source names and context.
- If the retrieved context is stronger for one report than the other, clearly say that.
- Do not use general knowledge.

User Comparison Question:
{question}

Retrieved Context:
{context}

Return:
1. Direct Comparison
2. Earlier Report / First Report Themes
3. Later Report / Second Report Themes
4. Common Themes Across Reports
5. Key Differences
6. Implications for Banks / Financial Institutions
7. Source References
""")

    chain = prompt | get_llm()
    response = chain.invoke({
        "question": comparison_question,
        "context": context
    })

    return response.content, docs

if uploaded_files:
    if st.button("Index Documents"):
        with st.spinner("Reading and indexing documents..."):
            docs = load_pdfs(uploaded_files)
            st.session_state.vector_store = create_vector_store(docs)
            st.session_state.chat_history = []
            st.session_state.uploaded_file_names = [file.name for file in uploaded_files]

        st.success("Documents indexed successfully.")
        st.write("Indexed files:")
        for name in st.session_state.uploaded_file_names:
            st.write(f"- {name}")

if st.session_state.vector_store:
    st.markdown("## Research Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Individual Report Summaries"):
            with st.spinner("Generating individual report summaries..."):
                individual_summaries = generate_individual_report_summaries(
                    st.session_state.vector_store,
                    st.session_state.uploaded_file_names
                )

            st.markdown("## Individual Report Summaries")

            for file_name, result in individual_summaries.items():
                st.markdown(f"### {file_name}")
                st.write(result["summary"])

                with st.expander(f"Source snippets for {file_name}"):
                    for i, doc in enumerate(result["docs"], start=1):
                        st.markdown(f"### Source {i}")
                        st.write(f"Document: {doc.metadata.get('source')}")
                        st.write(f"Page: {doc.metadata.get('page')}")
                        st.write(doc.page_content[:1000])

    with col2:
        if st.button("Combined Report Summary"):
            with st.spinner("Generating combined report summary..."):
                summary, summary_docs = generate_combined_report_summary(
                    st.session_state.vector_store
                )

            st.markdown("## Combined Report Summary")
            st.write(summary)

            with st.expander("Combined summary source snippets"):
                for i, doc in enumerate(summary_docs, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(f"Document: {doc.metadata.get('source')}")
                    st.write(f"Page: {doc.metadata.get('page')}")
                    st.write(doc.page_content[:1000])

    with col3:
        if st.button("Identify Key Risks"):
            with st.spinner("Identifying key risks..."):
                risks, risk_docs = identify_key_risks(
                    st.session_state.vector_store
                )

            st.markdown("## Key Risks")
            st.write(risks)

            with st.expander("Risk source snippets"):
                for i, doc in enumerate(risk_docs, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(f"Document: {doc.metadata.get('source')}")
                    st.write(f"Page: {doc.metadata.get('page')}")
                    st.write(doc.page_content[:1000])

    st.markdown("## Compare Reports")

    comparison_question = st.text_input(
        "Ask a comparison question",
        placeholder="Example: Compare the key risks highlighted across the uploaded reports."
    )

    if st.button("Compare Reports"):
        if comparison_question:
            with st.spinner("Comparing reports..."):
                comparison_answer, comparison_docs = compare_reports(
                    st.session_state.vector_store,
                    comparison_question
                )

            st.markdown("## Comparison Answer")
            st.write(comparison_answer)

            with st.expander("Comparison source snippets"):
                for i, doc in enumerate(comparison_docs, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(f"Document: {doc.metadata.get('source')}")
                    st.write(f"Page: {doc.metadata.get('page')}")
                    st.write(doc.page_content[:1000])
        else:
            st.warning("Please enter a comparison question first.")

    st.markdown("## Ask Questions")

    question = st.chat_input("Ask a specific question or follow-up question about the uploaded reports")

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Generating answer..."):
            answer, relevant_docs, standalone_question = answer_question(
                st.session_state.vector_store,
                question,
                st.session_state.chat_history
            )

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("Retrieval details"):
                st.write("Standalone retrieval question:")
                st.write(standalone_question)

            with st.expander("Retrieved source snippets"):
                for i, doc in enumerate(relevant_docs, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(f"Document: {doc.metadata.get('source')}")
                    st.write(f"Page: {doc.metadata.get('page')}")
                    st.write(doc.page_content[:1000])

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

else:
    st.info("Upload one or more PDFs and click 'Index Documents' to begin.")