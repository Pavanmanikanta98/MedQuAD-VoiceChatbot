import os
import time
import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import assemblyai as aai
from medQa_search import get_medlineplus_topic_summary

from dotenv import load_dotenv
load_dotenv()

# ---------- Set up AssemblyAI for Speech-to-Text ----------
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
transcriber = aai.Transcriber()

def speech_to_text(audio_file_path):
    """
    Convert a local audio file (e.g., WAV or MP3) to text using AssemblyAI.
    """
    transcript = transcriber.transcribe(audio_file_path)
    print("hello")
    if transcript.status == aai.TranscriptStatus.error:
        return f"Error: {transcript.error}"
    else:
        return transcript.text

# ---------- Load Fine-Tuned Model and Tokenizer ----------
peft_model_path = "./results-20250302T115210Z-001/results/t5-peft_model"
model = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# RAG





# ---------- Inference Function ----------
def infer_question(question_text):
    """Generate an answer for a given medical question using the fine-tuned model."""
    print("under the inference function")
    prompt = f"medical question: {question_text.strip()} answer:"
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

# ---------- Combined Function for Gradio Interface ----------
def voice_medical_qa(audio_input):
    """
    Takes a recorded audio file, converts it to text via AssemblyAI,
    generates an answer using the fine-tuned model,
    and returns the transcribed question and answer text.
    """
    # Convert speech to text using AssemblyAI
    transcribed_question = speech_to_text(audio_input)
    print("Transcribed")
    # Generate answer text using the fine-tuned model
    answer_text = infer_question(transcribed_question)
    
    return transcribed_question, answer_text

# ---------- Create Gradio Interface ----------
iface = gr.Interface(
    fn=voice_medical_qa,
    # inputs=gr.Audio(source="microphone", type="filepath", label="Record your question"),
    inputs = gr.Audio(sources=["microphone"], type="filepath", label="Record your question"),
  

    outputs=[
        gr.Textbox(label="Transcribed Question"),
        gr.Textbox(label="Model Answer")
    ],
    title="Medical QA Voice Chatbot",
    description="Speak your medical question and receive a text-based answer."
)

iface.launch(share=True)
