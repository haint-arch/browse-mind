import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os
import json
import io
import gradio as gr
import requests
import wave
import ffmpeg
import queue
import threading
from dotenv import load_dotenv
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import speech
from werkzeug.utils import secure_filename

load_dotenv()
app = Flask(__name__)
CORS(app)
sockets = Sockets(app)

logging.basicConfig(level=logging.INFO)

AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg/bin/ffprobe.exe"

api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sa-speech-history-browser.json'

history_data = []
sample_questions = [
    "Lịch sử duyệt web của tôi là gì?",
    "Hiển thị các tìm kiếm gần đây của tôi.",
    "Tôi đã truy cập những trang web nào tuần trước?",
    "Bạn có thể liệt kê lịch sử duyệt web của tôi không?",
    "Những trang web tôi đã truy cập gần đây là gì?"
]

# Lưu trữ embeddings cho history_data
vectorstore = None

# Load the tokenizer and model
model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
# model_name = 'keepitreal/vietnamese-sbert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute sentence embeddings and normalize them for cosine similarity
def get_normalized_embeddings(sentences, max_length=128):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    # Normalize the embeddings to use cosine similarity
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms 
    return normalized_embeddings.cpu().numpy()

# Compute embeddings for sample questions 
sample_question_embeddings = get_normalized_embeddings(sample_questions)

def transcribe_audio_stream(audio_generator):
    credentials = service_account.Credentials.from_service_account_file('sa-speech-history-browser.json')
    client = speech.SpeechClient(credentials=credentials)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='vi-VN'
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    for response in responses:
        for result in response.results:
            if result.is_final:
                logging.info(f"Transcript: {result.alternatives[0].transcript}")
                yield result.alternatives[0].transcript

# Function to check if a question is within the scope of the website titles
def is_question_within_scope(question, sample_questions, sample_question_embeddings):
    # Compute embedding for the user's question
    question_embedding = get_normalized_embeddings([question])
    
    # Use FAISS to find the most similar sample question
    index = faiss.IndexFlatIP(sample_question_embeddings.shape[1])
    index.add(sample_question_embeddings)
    
    distances, indices = index.search(question_embedding, k=1)
    
    # Check if the most similar sample question is within a certain threshold
    logging.info("Similarity score for the user's query: %.2f", distances[0][0])
    if distances[0][0] > 0.1:
        return True, sample_questions[indices[0][0]]
    else:
        return False, "Câu hỏi không liên quan đến lịch sử duyệt web."

# Ensure data is a list of dictionaries with 'title' and 'url'
def validate_history_data(data):
    if not isinstance(data, list) or not all(isinstance(item, dict) and 'title' in item and 'url' in item for item in data):
        raise ValueError("Data must be a list of dictionaries with 'title' and 'url'")

# Function to compute and cache embeddings for history_data
def compute_history_embeddings():
    global vectorstore
    titles = [item['title'] for item in history_data]
    embeddings = get_normalized_embeddings(titles)
    vectorstore = faiss.IndexFlatIP(embeddings.shape[1])
    vectorstore.add(embeddings)
    logging.info("Embeddings for history data computed and stored in vectorstore.")

# Function to search for the most similar response
def search(query):
    logging.info("Searching for: %s with history length: %d", query, len(history_data))
    if not history_data:
        return None, 0.0

    validate_history_data(history_data)

    if vectorstore is None:
        logging.info("Vectorstore is not initialized. Computing embeddings for history data.")
        compute_history_embeddings()

    query_embedding = get_normalized_embeddings([query])

    logging.info("Searching for the most similar response.")
    distances, indices = vectorstore.search(query_embedding, k=1)
    return history_data[indices[0][0]], distances[0][0]


# API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('query', '')
    logging.info("Received user message: %s", user_message)
    
    is_within_scope, relevant_title = is_question_within_scope(user_message, sample_questions, sample_question_embeddings)
    logging.info("Is question within scope: %s", is_within_scope)

    if is_within_scope:
        result, similarity = search(user_message)
        if result:
            response = f"Câu tiêu đề liên quan nhất: {result['title']} (Độ tương đồng: {similarity:.2f}). URL: {result['url']}"
        else:
            response = "Không tìm thấy tiêu đề liên quan."
    else:
        response = "Câu hỏi này không liên quan đến tiêu đề website."
    
    return jsonify({'response': response})

@sockets.route('/transcribe_stream')
def transcribe_socket(ws):
    audio_queue = queue.Queue()

    def audio_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def receive_audio():
        while not ws.closed:
            message = ws.receive()
            if message:
                audio_queue.put(message)
            else:
                audio_queue.put(None)
                break

    threading.Thread(target=receive_audio).start()

    for transcript in transcribe_audio_stream(audio_generator()):
        ws.send(json.dumps({"transcription": transcript}))

@app.route('/upload_history', methods=['POST'])
def upload_history():
    global history_data, vectorstore
    new_history_data = request.json.get('history', [])
    
    try:
        validate_history_data(new_history_data)
    except ValueError as e:
        logging.error("Validation error: %s", e)
        return jsonify({'status': 'error', 'message': str(e)}), 400
    
    # Append new history data to existing history_data
    history_data.extend(new_history_data)
    
    # Compute embeddings for new history data and add to vectorstore
    new_titles = [item['title'] for item in new_history_data]
    logging.info("New history data uploaded: %d items", len(new_titles))

    # Check if new_titles is empty
    if not new_titles:
        logging.warning("No new titles to embed.")
        return jsonify({'status': 'success', 'message': 'No new titles to embed.'})

    new_embeddings = get_normalized_embeddings(new_titles)
    if vectorstore is None:
        vectorstore = faiss.IndexFlatIP(new_embeddings.shape[1])
    vectorstore.add(new_embeddings)
    logging.info("New history data uploaded and embeddings computed.")
    
    return jsonify({'status': 'success'})
    
if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    logging.info("Starting server on port 5000")
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()