import os
import time
import torch
import gradio as gr
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
# from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore import InMemoryDocstore

from langchain.prompts import ChatPromptTemplate  # prompt template

import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()

# ---------- (Voice code commented out; using text input) ----------
# aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
# transcriber = aai.Transcriber()
# def speech_to_text(audio_file_path):
#     transcript = transcriber.transcribe(audio_file_path)
#     if transcript.status == aai.TranscriptStatus.error:
#         return f"Error: {transcript.error}"
#     else:
#         return transcript.text

# ---------- Load Fine-Tuned Model and Tokenizer ----------
peft_model_path = "./results-20250302T115210Z-001/results/t5-peft_model"
model = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------- Embedding and FAISS Vector Store Setup using LangChain ----------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create an empty document store using InMemoryDocstore
docstore = InMemoryDocstore({})
documents = []  
embedding_dim = 768  # all-mpnet-base-v2 produces 768-dimensional embeddings
import faiss
empty_index = faiss.IndexFlatL2(embedding_dim)
vector_store = LCFAISS(embeddings, empty_index, docstore, {})  # initialize with a docstore that supports adding

def add_document_to_index(doc_text):
    """
    Add a new text to the vector store using the add_texts method.
    """
    vector_store.add_texts([doc_text])
    print("Added new text to vector store.")

# ---------- Import MedlinePlus Retrieval Function from Module ----------
from medQa_search import get_medlineplus_topic_summary

# ---------- Wikipedia Retrieval using LangChain Community Tools ----------
try:
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
    wiki_query = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

    def get_wikipedia_topic_summary(query):
        """Retrieve a brief snippet from Wikipedia based on the query."""
        try:
            context = wiki_query.run(query)
            return context
        except Exception as e:
            print("Wikipedia retrieval error:", e)
            return ""
except ImportError:
    print("langchain_community not installed. Wikipedia retrieval disabled.")
    def get_wikipedia_topic_summary(query):
        return ""

# ---------- Define Prompt Template ----------
prompt_template = ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        If the question requests a diagnosis or treatment recommendation, advise consulting a local doctor.
        Context:
        {context}

        Question:
        {input}
    """
)

# ---------- Combined Retrieval Function (Individual Fallbacks) ----------
def retrieve_context(query, similarity_threshold=0.5):
    """
    Retrieve relevant context from the vector store.
    If no document is similar enough to the query,
    fall back to retrieving context from MedlinePlus and Wikipedia separately,
    add each text individually to the vector store using add_texts,
    then perform a similarity search and return the best match.
    """
    docs_and_scores = vector_store.similarity_search_with_score(query, k=1)
    if docs_and_scores:
        best_doc, best_score = docs_and_scores[0]
        similarity = 1 / (1 + best_score)  # similarity = 1 / (1 + L2 distance)
        print(f"Initial best similarity score: {similarity:.2f}")
        if similarity >= similarity_threshold:
            print("Retrieved context from vector store.")
            return best_doc.page_content

    # Fallback: Retrieve contexts separately.
    medline_context = get_medlineplus_topic_summary(query)
    wiki_context = get_wikipedia_topic_summary(query)
    
    if medline_context:
        # print("Retrieved MedlinePlus context:")
        # print(medline_context)
        add_document_to_index(medline_context)
    if wiki_context:
        # print("Retrieved Wikipedia context:")
        # print(wiki_context)
        add_document_to_index(wiki_context)
    
    docs_and_scores = vector_store.similarity_search_with_score(query, k=1)
    if docs_and_scores:
        best_doc, best_score = docs_and_scores[0]
        similarity = 1 / (1 + best_score)
        print(f"New best similarity score: {similarity:.2f}")
        if similarity >= similarity_threshold - 0.15:
            print("Retrieved context after adding fallback texts.")
            return best_doc.page_content
    return ""

# ---------- Inference Function with Prompt Template ----------
def infer_question_with_context(query_text):
    """
    Generate an answer for a given medical question using the fine-tuned model,
    augmented with context retrieved from the vector store.
    The prompt is generated using a ChatPromptTemplate.
    """
    external_context = retrieve_context(query_text)
    print("Retrieved context:", external_context)
    
    print("------------------prompt templete working------------------------")
    prompt = prompt_template.format(context=external_context, input=query_text.strip())
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ---------- Gradio Interface Function (Text Input) ----------
def text_medical_qa(question_text):
    """
    Accepts a typed medical question,
    retrieves external context using the vector store (with fallback to MedlinePlus and Wikipedia),
    generates an answer using the fine-tuned model,
    and returns the question and the model answer.
    """
    print("Received Question:", question_text)
    answer_text = infer_question_with_context(question_text)
    return question_text, answer_text

# ---------- Launch Gradio Interface with Text Input ----------
iface = gr.Interface(
    fn=text_medical_qa,
    inputs=gr.Textbox(lines=2, placeholder="Type your medical question here...", label="Medical Question"),
    outputs=[
        gr.Textbox(label="Your Question"),
        gr.Textbox(label="Model Answer")
    ],
    title="Medical QA Chatbot with Dynamic Retrieval & FAISS",
    description="Type your medical question. The system checks its vector DB (using HuggingFaceEmbeddings via LangChain FAISS) for relevant context. If not found, it retrieves data from MedlinePlus and Wikipedia individually, updates its DB, and uses that context to generate an answer."
)

iface.launch()
