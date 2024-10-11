from flask import Flask, request, jsonify
from flask_cors import CORS
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
from dotenv import load_dotenv
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import speech
from werkzeug.utils import secure_filename

load_dotenv()
app = Flask(__name__)
CORS(app)

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

def transcribe_audio(audio_file):
    credentials = service_account.Credentials.from_service_account_file('sa-speech-history-browser.json')
    client = speech.SpeechClient(credentials=credentials)

    # Load audio data
    audio_data = audio_file.read()

    # Convert stereo to mono using ffmpeg-python
    input_audio = ffmpeg.input('pipe:0')
    output_audio = ffmpeg.output(input_audio, 'pipe:1', ac=1, format='wav')
    process = ffmpeg.run_async(output_audio, pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    mono_audio_data, _ = process.communicate(input=audio_data)

    # Convert mono_audio_data (bytes) to a file-like object
    mono_audio_data_io = io.BytesIO(mono_audio_data)
    
    with wave.open(mono_audio_data_io, 'rb') as wf:
        sample_rate = wf.getframerate()

    audio = speech.RecognitionAudio(content=mono_audio_data_io.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code='vi-VN'  # For Vietnamese
    )

    # Send the request to Google Cloud
    response = client.recognize(config=config, audio=audio)

    # Extract and return the transcription
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
        return result.alternatives[0].transcript

# Function to check if a question is within the scope of the website titles
def is_question_within_scope(question, website_titles):
    # Craft the prompt
    prompt = f"Is the following question related to any of these website titles? If yes, provide the most relevant title. Question: '{question}'. Titles: {website_titles}. Remember to answer 'Yes' or 'No'."
    
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
            if "Yes" in content:
                return True, content
            else:
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

# Load the tokenizer and model
model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
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

# Load data from JSON file
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['history']

# Ensure data is a list of dictionaries with 'title' and 'url'
if not isinstance(data, list) or not all(isinstance(item, dict) and 'title' in item and 'url' in item for item in data):
    raise ValueError("Data must be a list of dictionaries with 'title' and 'url'")

# Extract titles for embedding
titles = [item['title'] for item in data]

# Convert data to normalized embeddings
embeddings = get_normalized_embeddings(titles)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Function to search for the most similar response
def search(query):
    query_embedding = get_normalized_embeddings([query])
    distances, indices = index.search(query_embedding, k=1)
    return data[indices[0][0]], distances[0][0]

# API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('query', '')
    
    is_within_scope, relevant_title = is_question_within_scope(user_message, data)
    
    if is_within_scope:
        result, similarity = search(user_message)
        response = f"Câu tiêu đề liên quan nhất: {result['title']} (Độ tương đồng: {similarity:.2f}). URL: {result['url']}"
    else:
        response = "Câu hỏi này không liên quan đến tiêu đề website."
    
    return jsonify({'response': response})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    transcription = transcribe_audio(audio_file)
    return jsonify({"transcription": transcription})

if __name__ == '__main__':
    app.run(debug=True)