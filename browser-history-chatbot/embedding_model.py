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
import re
import gradio as gr
import requests
import wave
import ffmpeg
import queue
import ast
import threading
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
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

# Lưu trữ embeddings cho history_data
vectorstore = None

# Load the tokenizer and model for semantic similarity
semantic_model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
semantic_model = AutoModel.from_pretrained(semantic_model_name)

def create_temp_vectorstore(filtered_history):
    temp_vectorstore = faiss.IndexFlatIP(vectorstore.d)
    temp_embeddings = get_normalized_embeddings([item['title'] for item in filtered_history])
    temp_vectorstore.add(temp_embeddings)
    return temp_vectorstore

def crawl_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)  # Lấy toàn bộ nội dung văn bản của trang web
        return content
    except requests.exceptions.RequestException as e:
        logging.error(f"Không thể truy cập URL {url}. Lỗi: {e}")
        return None

def extract_json_from_response(response_text):
    # Sử dụng biểu thức chính quy để trích xuất nội dung JSON từ chuỗi văn bản
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")
            return None
    return None

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

def get_normalized_embeddings(sentences, max_length=128):
    inputs = semantic_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = semantic_model(**inputs).pooler_output
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return (embeddings / norms).cpu().numpy()

def semantic_search(user_input, temp_vectorstore, filtered_history):
    query_embedding = get_normalized_embeddings([user_input])
    
    distances, indices = temp_vectorstore.search(query_embedding, k=1)
    return filtered_history[indices[0][0]], distances[0][0]

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

def classify_website(title, content):
    prompt = f"""
        Classify the website into one or more of the following categories: entertainment, music, news, social media, education, tech, health, shopping, finance, travel, productivity, forums, sports, food, science, home.

        Title: {title}
        Content: {content}

        Output the categories as a array.

        Example:
        Title: Example Title
        Content: Example content about a tech and productivity tool.
        Output: ["tech", "productivity"]
    """
    
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  
        result = response.json()
        
        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            match = re.search(r'\[.*?\]', content)
            if match:
                categories = match.group(0)
                result = json.loads(categories)
                valid_categories = []
                for category in result:
                    if category in ["entertainment", "music", "news", "social media", "education", "tech", "health", "shopping", "finance", "travel", "productivity", "forums", "sports", "food", "science", "home"]:
                        valid_categories.append(category)
                    else:
                        logging.error("Invalid category: %s", category)
                if valid_categories:
                    return valid_categories
                else:
                    return ["other"]
        else:
            logging.error("Unexpected API response format.")
            return ["other"]

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err} in API response")
    
    return ["other"]

def extract_features(input_text):
    prompt = f"""
        Extract the following features from the text provided, returning only the output in JSON format:

        1. **time** - Look for references to times or events that may indicate when the action or request occurred, such as "yesterday at 7 PM" or "3 days ago." Normalize the time to obtain `start_date` and `end_date` in a standard format to support time constraints. 

        - For example, if the text includes "2 days ago," calculate `start_date` as two days before today’s date (today is 31/10/2024): 
            - `start_date`: 00:00 on 29/10/2024
            - `end_date`: 23:59 on 29/10/2024.
        
        - If the text specifies a time, like "today at 8 PM," convert to an hour range around the specified time:
            - `start_date`: 19:00 on 31/10/2024
            - `end_date`: 21:00 on 31/10/2024.

        2. **title** - Identify any references to topics or main themes, especially if they are phrases that start with or include "related to" or "main topic." For example, "related to food," "main theme about travel."

        3. **color** - Detect references to primary or dominant colors mentioned in the text, such as "yellow," "blue," or "green."

        4. **content** - Extract the main content or focus area of the text, often appearing after phrases like "main content about" or "content regarding." For example, "main content about Vietnamese bun cha dish."

        5. **category** - Classify the website into one of the following categories: entertainment, music, news, social media, education, tech, health, shopping, finance, travel, productivity, forums, sports, food, science, home.

        Example Input: "Tìm cho tôi 1 website tôi có truy cập vào 7h tối qua, web này có tiêu đề liên quan đến ẩm thực và màu chủ đạo là màu vàng, nội dung chính về món ăn bún chả Việt Nam."

        Expected JSON Output:
        {{
        "time": {{
            "original": "7h tối qua",
            "value": "2024-10-30T19:00:00",
            "start_date": "2024-10-30T18:00:00",
            "end_date": "2024-10-30T20:00:00"
        }},
        "title": "liên quan đến ẩm thực",
        "color": "màu vàng",
        "content": "món ăn bún chả Việt Nam",
        "category": "food"
        }}

        Text to Analyze: "{input_text}"

        Output JSON Only:
        """
        
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  
        result = response.json()
        
        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            return content
        else:
            print("Unexpected API response format.")
            return None

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err} in API response")
    
    return None

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

    if is_within_scope:
        features_response = extract_features(user_message)
        logging.info("Extracted features response: %s", features_response)

        features = extract_json_from_response(features_response)
        logging.info("Extracted features: %s", features)

        if features:
            filtered_history = history_data
            # Thời gian 
            if 'time' in features:
                search_time = datetime.strptime(features['time']['value'], "%Y-%m-%dT%H:%M:%S")
                start_time = datetime.strptime(features['time']['start_date'], "%Y-%m-%dT%H:%M:%S")
                end_time = datetime.strptime(features['time']['end_date'], "%Y-%m-%dT%H:%M:%S")

                filtered_history = [item for item in history_data if start_time.timestamp() <= item['lastVisitTime'] / 1000 <= end_time.timestamp()]

            # Tiêu đề liên quan nhất
            if 'title' in features:
                title_content = features['title']
                temp_vectorstore = create_temp_vectorstore(filtered_history)
                result, similarity = semantic_search(title_content, temp_vectorstore, filtered_history)

                if result:
                    response = (f"Câu tiêu đề liên quan nhất: {result['title']} (Độ tương đồng: {similarity:.2f}). "
                                f"URL: {result['url']}. Từ khóa: {features}")
                else:
                    response = "Không tìm thấy tiêu đề liên quan."

            # Nội dung liên quan nhất
            if 'content' in features:
                content = features['content']

            # Màu sắc liên quan nhất
            if 'color' in features:
                color = features['color']
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

def save_history_to_json(new_history_data):
    file_path = 'history_data.json'
    if os.path.exists(file_path):
        # Đọc dữ liệu hiện có từ file
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Kết hợp dữ liệu hiện có với dữ liệu mới
    combined_data = existing_data + new_history_data

    # Lưu dữ liệu kết hợp trở lại file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
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

    # Crawl nội dung từ các trang web trong lịch sử
    for item in new_history_data:
        content = crawl_website_content(item['url'])
        item['content'] = content

        # Phân loại website dựa trên nội dung và tiêu đề
        categories = classify_website(item['title'], content)
        item['categories'] = categories

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