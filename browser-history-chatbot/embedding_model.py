import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification, pipeline
from urllib.parse import urlparse, parse_qs, urlunparse
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
from pyvi.ViTokenizer import tokenize

logging.info(f"NumPy version: {np.__version__}")

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

# Lưu trữ embeddings cho history_data
vectorstore = None

# Load the tokenizer and model for semantic similarity
semantic_model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
semantic_model = AutoModel.from_pretrained(semantic_model_name)

# Load a lightweight NER model (DistilBERT)
ner_model_name = "distilbert-base-uncased"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

def normalize_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'v' in query_params:
        normalized_query = f"v={query_params['v'][0]}"
        normalized_url = urlunparse(parsed_url._replace(query=normalized_query))
        return normalized_url
    return url

def filterDuplicateHistory(historyItems):
    uniqueHistory = []
    seenUrls = set()
    seenTitles = set()

    for item in historyItems:
        normalized_url = normalize_url(item['url'])
        if normalized_url not in seenUrls and item['title'] not in seenTitles:
            uniqueHistory.append(item)
            seenUrls.add(normalized_url)
            seenTitles.add(item['title'])
    return uniqueHistory

# Hàm tách từ khóa từ câu hỏi người dùng
def extract_keywords(prompt):
    ner_results = ner_pipeline(prompt)
    keywords = [entity['word'] for entity in ner_results if entity['entity'] in ["TITLE", "CONTENT", "COLOR", "TIME"]]
    return " ".join(keywords)

def get_normalized_embeddings(sentences, max_length=128):
    sentences = [tokenize(sentence) for sentence in sentences]
    inputs = semantic_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = semantic_model(**inputs).last_hidden_state.mean(dim=1)
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return (embeddings / norms).cpu().numpy()

def semantic_search_with_ner(user_input):
    keywords = extract_keywords(user_input)
    logging.info("Extracted Keywords: %s", keywords)
    query_embedding = get_normalized_embeddings([keywords])
    
    if vectorstore is None:
        logging.info("Vectorstore is not initialized. Computing embeddings for history data.")
        compute_history_embeddings()
    
    distances, indices = vectorstore.search(query_embedding, k=1)
    return history_data[indices[0][0]], distances[0][0]

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
def is_question_within_scope(question):
    # Craft the prompt
    prompt = f"Is the following question within the scope of browser history search?\n\nQuestion: {question}\n\nAnswer with 'yes' or 'no'."
    
    # Prepare the payload for the API request
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  
        result = response.json()
        
        # Print the full response for debugging
        print("API Response:", result)
        
        # Check if the expected fields are in the response
        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            if "Yes" in content or "yes" in content:	
                return True, content
            elif "No" in content or "no" in content:
                return False, content
        else:
            print("Unexpected API response format.")
            return False, "Unexpected API response."

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")
    
    return False, "API request failed."

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
    
    is_within_scope, relevant_title = is_question_within_scope(user_message)
    logging.info("Is question within scope: %s", is_within_scope)

    if is_within_scope:
        result, similarity, entities = semantic_search_with_ner(user_message)
        if result:
            response = (f"Câu tiêu đề liên quan nhất: {result['title']} (Độ tương đồng: {similarity:.2f}). "
                        f"URL: {result['url']}. Từ khóa: {entities}")
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

def save_history_to_json(history_data):
    with open('history_data.json', 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)
    logging.info("History data saved to history_data.json")

@app.route('/upload_history', methods=['POST'])
def upload_history():
    global history_data, vectorstore
    new_history_data = request.json.get('history', [])
    
    try:
        validate_history_data(new_history_data)
    except ValueError as e:
        logging.error("Validation error: %s", e)
        return jsonify({'status': 'error', 'message': str(e)}), 400

    # Lọc dữ liệu lịch sử để loại bỏ các URL và tiêu đề trùng lặp
    new_history_data = filterDuplicateHistory(new_history_data)
    
    # Append new history data to existing history_data
    history_data.extend(new_history_data)

    # Save history data to JSON file
    save_history_to_json(history_data)
    
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