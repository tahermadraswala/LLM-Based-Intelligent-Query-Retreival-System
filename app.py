# app.py
import gradio as gr
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Title
title = "PDF Intelligent Query System"
description = "Upload any PDF and ask questions – powered by RAG (TinyLlama + FAISS)"

# Global state
vectorstore = None
generator = None

def load_llm():
    global generator
    if generator is None:
        MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return generator

RAG_PROMPT = """<|system|>
You are a helpful assistant. Answer using only the provided context.
</|system|>
<|user|>
Context:
{context}

Question: {question}
</|user|>
<|assistant|>
"""
prompt_template = PromptTemplate.from_template(RAG_PROMPT)

def process_pdf(file):
    global vectorstore
    if file is None:
        return "No file uploaded.", gr.update(visible=False)

    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.read())
        pdf_path = f.name

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]

        embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts, embedder)
        os.unlink(pdf_path)
        return f"Success: Processed {len(chunks)} chunks. Ask a question!", gr.update(visible=True)
    except Exception as e:
        os.unlink(pdf_path)
        return f"Error: {str(e)}", gr.update(visible=False)

def answer_question(question, history):
    global vectorstore, generator
    if vectorstore is None or not question.strip():
        return history + [(question, "Upload and process a PDF first.")], ""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    formatted = prompt_template.format(context=context, question=question)

    load_llm()
    output = generator(formatted)[0]["generated_text"]
    answer = output.split("<|assistant|>")[-1].strip()

    # Sources
    sources = []
    for i, d in enumerate(docs, 1):
        page = int(d.metadata.get("page", 0)) + 1
        snippet = d.page_content[:200] + ("..." if len(d.page_content) > 200 else "")
        sources.append(f"**[{i}] Page {page}:** {snippet}")

    full_answer = f"{answer}\n\n**Sources:**\n" + "\n".join(sources)
    history.append((question, full_answer))
    return history, ""

# Gradio Interface
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n{description}")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Ask a question", placeholder="e.g., What is the filing deadline?")
            send_btn = gr.Button("Send")

    process_btn.click(process_pdf, inputs=pdf_file, outputs=[status, chatbot])
    send_btn.click(answer_question, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(answer_question, inputs=[msg, chatbot], outputs=[chatbot, msg])

demo.launch()
