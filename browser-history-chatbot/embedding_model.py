from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os
import json
import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    data = json.load(f)['data']

# Ensure data is a list of strings
if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
    raise ValueError("Data must be a list of strings")

# Convert data to normalized embeddings
embeddings = get_normalized_embeddings(data)

# Function to search for the most similar response
def search(query):
    query_embedding = get_normalized_embeddings([query])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(query_embedding, k=1)
    return data[indices[0][0]], distances[0][0]

# API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('query', '')
    
    is_within_scope, relevant_title = is_question_within_scope(user_message, data)
    
    if is_within_scope:
        result, similarity = search(user_message)
        response = f"Câu tiêu đề liên quan nhất: {result} (Độ tương đồng: {similarity:.2f})"
    else:
        response = "Câu hỏi này không liên quan đến tiêu đề website."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)