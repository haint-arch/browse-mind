import json
import logging
import os
import queue
import re
import time
import threading
from collections import Counter, deque
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse, parse_qs, urlunparse

import wave
import io
import ffmpeg
import faiss
import numpy as np
import requests
import torch
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from geventwebsocket.websocket import WebSocket
from google.cloud import speech
from google.oauth2 import service_account
from PIL import Image
from bs4 import BeautifulSoup
from colorthief import ColorThief
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets
from pydub import AudioSegment
from transformers import AutoModel, AutoTokenizer

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
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sa-speech-history-browser.json'

history_buffer = deque(maxlen=20)  # Tăng kích thước buffer
last_process_time = time.time()
BUFFER_PROCESS_INTERVAL = 300  # Process buffer every 5 minutes
MIN_ITEMS_TO_PROCESS = 5  # Số lượng tối thiểu các mục để xử lý

title_vectorstore = None
content_vectorstore = None
history_data = []

# Load the tokenizer and model for semantic similarity
semantic_model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
semantic_model = AutoModel.from_pretrained(semantic_model_name)


# Methods for color extraction
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def parse_color(color):
    if not color or not isinstance(color, str):
        return None
    color = color.strip()
    if not color:
        return None

    try:
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            if len(hex_color) == 3:  # Convert #RGB to #RRGGBB
                hex_color = ''.join(c + c for c in hex_color)
            if len(hex_color) != 6:  # Handle invalid hex string length
                raise ValueError(f"Invalid hex color length: {hex_color}")
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        elif color.startswith('rgb'):
            values = re.findall(r'\d+', color)
            if len(values) == 3:
                return tuple(map(int, values))
        else:
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255),
            }
            return color_map.get(color.lower())
    except (ValueError, IndexError) as e:
        logging.warning(f"Error parsing color '{color}': {str(e)}")
        return None
    return None


def pool_colors(colors, threshold=20):
    pooled = []
    for color in colors:
        for i, pooled_color in enumerate(pooled):
            if all(abs(c1 - c2) <= threshold for c1, c2 in zip(color, pooled_color)):
                pooled[i] = tuple((c1 + c2) // 2 for c1, c2 in zip(color, pooled_color))
                break
        else:
            pooled.append(color)
    return pooled


def get_css_colors(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        inline_styles = [tag.get('style', '') for tag in soup.find_all(style=True)]
        stylesheets = [link.get('href', '') for link in soup.find_all('link', rel='stylesheet')]

        colors = []

        for style in inline_styles:
            if style:
                colors.extend(re.findall(r'#[0-9a-fA-F]{3,6}|rgb$$\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$$', style))

        for stylesheet in stylesheets:
            if not stylesheet:
                continue
            if not stylesheet.startswith(('http:', 'https:')):
                stylesheet = f"{url.rstrip('/')}/{stylesheet.lstrip('/')}"
            try:
                css_response = requests.get(stylesheet, headers=HEADERS, timeout=5)
                css_response.raise_for_status()
                colors.extend(re.findall(r'#[0-9a-fA-F]{3,6}|rgb$$\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$$', css_response.text))
            except Exception as e:
                logging.warning(f"Error fetching stylesheet {stylesheet}: {str(e)}")
                continue

        parsed_colors = []
        for color in colors:
            parsed = parse_color(color)
            if parsed:
                parsed_colors.append(parsed)

        return parsed_colors
    except Exception as e:
        logging.error(f"Error getting CSS colors from {url}: {str(e)}")
        return []


def get_image_from_url(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error fetching image from {url}: {str(e)}")
        return None


def get_dominant_color_from_image(image):
    if image is None:
        return None
    try:
        # Chuyển đổi hình ảnh sang định dạng RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Lưu hình ảnh tạm thời vào bộ nhớ
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Sử dụng ColorThief để lấy màu chủ đạo
        color_thief = ColorThief(img_byte_arr)
        return color_thief.get_color(quality=1)
    except Exception as e:
        print(f"Error getting dominant color from image: {str(e)}")
        return None


def get_dominant_color(url, default_color=(255, 255, 255)):
    try:
        # Get colors from CSS
        css_colors = get_css_colors(url)
        if css_colors:
            pooled_css_colors = pool_colors([c for c in css_colors if c is not None])
            if pooled_css_colors:
                css_color_counts = Counter(pooled_css_colors)

                # Try to get image color
                try:
                    image = get_image_from_url(url)
                    image_color = get_dominant_color_from_image(image) if image else None

                    # Combine results
                    all_colors = list(css_color_counts.keys())
                    if image_color:
                        all_colors.append(image_color)

                    pooled_all_colors = pool_colors(all_colors)
                    if pooled_all_colors:
                        return Counter(pooled_all_colors).most_common(1)[0][0]
                except Exception as e:
                    logging.warning(f"Error processing image from {url}: {str(e)}")
                    # Fall back to CSS colors only
                    return css_color_counts.most_common(1)[0][0]

        # If no CSS colors found, try image only
        image = get_image_from_url(url)
        if image:
            image_color = get_dominant_color_from_image(image)
            if image_color:
                return image_color

        logging.warning(f"No colors found for {url}, using default color")
        return default_color
    except Exception as e:
        logging.error(f"Error processing {url}: {str(e)}")
        return default_color


# Methods for website content extraction
def crawl_website_content(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = " ".join([p.get_text() for p in soup.find_all('p')])

        if not main_content:
            logging.warning("No main content found in URL: %s", url)
            return None, []

        summary = extract_summary(main_content)
        if not summary:
            logging.warning("Failed to extract summary from URL: %s", url)
            return None, []

        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary)
        return summary, sentences
    except requests.exceptions.RequestException as e:
        logging.error(f"Không thể truy cập URL {url}. Lỗi: {e}")
        return None, []


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


# Methods for semantic search
def normalize_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'v' in query_params:
        normalized_query = f"v={query_params['v'][0]}"
        normalized_url = urlunparse(parsed_url._replace(query=normalized_query))
        return normalized_url
    return url


def filter_duplicate_history(history_items):
    """
    Lọc các URL hoặc tiêu đề trùng lặp từ lịch sử duyệt web.
    """
    unique_history = []
    seen_urls = set()
    seen_titles = set()

    for item in history_items:
        normalized_url = normalize_url(item['url'])
        if normalized_url not in seen_urls and item['title'] not in seen_titles:
            unique_history.append(item)
            seen_urls.add(normalized_url)
            seen_titles.add(item['title'])
    return unique_history


def get_normalized_embeddings(sentences, max_length=128):
    if not sentences:
        raise ValueError("The input sentences list is empty.")
    inputs = semantic_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = semantic_model(**inputs).pooler_output
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return (embeddings / norms).cpu().numpy()


def semantic_search(query_embedding, vectorstore, filtered_history, k=1):
    distances, indices = vectorstore.search(query_embedding, k=k)
    results = [
        (filtered_history[i], float(distances[0][j]))
        for j, i in enumerate(indices[0])
    ]
    return results


# Methods for prompt-based API calls
def extract_summary(input_text):
    prompt = f"""
        Summarize the following content in vietnamese:

        {input_text}

        Output the result only.
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
            return content.strip()
        else:
            logging.error("Unexpected API response format.")
            return None

    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        logging.error(f"Error occurred: {err}")
    except KeyError as err:
        logging.error(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        logging.error(f"JSONDecodeError: {err} in API response")

    return None


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
                    if category in ["entertainment", "music", "news", "social media", "education", "tech", "health",
                                    "shopping", "finance", "travel", "productivity", "forums", "sports", "food",
                                    "science", "home"]:
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
    current_time = datetime.now()
    time_now = current_time.strftime("%Y-%m-%dT%H:%M:%S")
    time_one_month_ago = (current_time - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")

    prompt = f"""
        Extract the following features from the text provided, returning only the output in JSON format:

        1. **time** - Analyze references to time or events in the text and normalize them to a standard format. Use Vietnam Time Zone (GMT+7). Interpret time-related phrases as follows:
           - "gần đây nhất" (most recently): 
             - `value`: null
             - `start_date`: {time_one_month_ago}
             - `end_date`: {time_now}
             - `rank`: 1
           - "gần đây thứ hai" (second most recently):
             - `value`: null
             - `start_date`: {time_one_month_ago}
             - `end_date`: {time_now}
             - `rank`: 2
           - "xa nhất" (farthest):
             - `value`: null
             - `start_date`: "2020-01-01T00:00:00"
             - `end_date`: {time_one_month_ago}
             - `rank`: -1
           - For specific time references like "hôm qua 7h tối" (yesterday at 7 PM) and today is {time_now}:
             - Calculate the exact time range
             - `value`: "2021-07-01T19:00:00"
             - `start_date`: "2021-07-01T18:00:00"
             - `end_date`: "2021-07-01T20:00:00"
             - `rank`: -1
             - Set `rank` to 0 (indicating a specific time, not a relative ranking)
           - For vague terms like "gần đây" (recently):
             - `value`: null
             - `start_date`: {time_one_month_ago}
             - `end_date`: {time_now}
             - `rank`: 0

        2. **title** - Extract phrases related to the main topic, often appearing after "liên quan đến" (related to) or "chủ đề chính là" (main topic is).

        3. **color** - Detect mentions of colors (e.g., "vàng" (yellow), "xanh" (blue), "đỏ" (red)). Return in RGB format.

        4. **content** - Extract the core focus or topic, often found after "nội dung chính là" (main content is) or "về" (about).

        5. **category** - Classify into: entertainment, music, news, social media, education, tech, health, shopping, finance, travel, productivity, forums, sports, food, science, home.

        Example Input: "Tìm cho tôi website truy cập gần đây nhất có tiêu đề liên quan đến công nghệ và màu chủ đạo là xanh lá cây."

        Expected JSON Output:
        {{
          "time": {{
            "original": "gần đây nhất",
            "value": null,
            "start_date": "{time_one_month_ago}",
            "end_date": "{time_now}",
            "rank": 1
          }},
          "title": "liên quan đến công nghệ",
          "color": [0, 255, 0],
          "content": null,
          "category": "tech"
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


# Methods for calculating scores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_time_score(query_time, page_time, max_time_diff):
    time_diff = abs(query_time - page_time)
    return 1 - (time_diff / max_time_diff)


def calculate_color_score(query_color, page_color):
    if not query_color or not page_color:
        return 0
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(query_color, page_color)))
    max_distance = np.sqrt(3 * (255 ** 2))
    return 1 - (distance / max_distance)


def calculate_category_score(query_category, page_categories):
    return 1 if query_category in page_categories else 0


def calculate_total_score(title_score, content_score, time_score, color_score, category_score):
    return (0.4 * title_score) + (0.4 * content_score) + (0.1 * time_score) + (0.05 * color_score) + (
            0.05 * category_score)


# API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    global history_data, title_vectorstore, content_vectorstore

    user_message = request.json.get('query', '')
    logging.info("Received user message: %s", user_message)

    is_within_scope, relevant_title = is_question_within_scope(user_message)

    if is_within_scope:
        features_response = extract_features(user_message)
        logging.info("Extracted features response: %s", features_response)

        features = extract_json_from_response(features_response)
        if features:
            query_embedding = get_normalized_embeddings([user_message])

            # Initialize lists to store top results for each feature
            top_results = {
                'title': [],
                'content': [],
                'time': [],
                'color': [],
                'category': []
            }

            # Process each feature and collect top results
            if features.get('title'):
                top_results['title'] = semantic_search(query_embedding, title_vectorstore, history_data, k=10)

            if features.get('content'):
                top_results['content'] = semantic_search(query_embedding, content_vectorstore, history_data, k=10)

            if features.get('time') and (features['time'].get('value') or features['time'].get('start_date')) and features['time'].get('rank') and features['time'].get('original'):
                rank = features['time']['rank']
                # Calculate time scores for all top items
                # For non-specific time queries
                if rank == 0:
                    dt = datetime.strptime(features['time']['value'], '%Y-%m-%dT%H:%M:%S')
                    query_time = int(dt.timestamp()) * 1000

                    time_results = [
                        (item, calculate_time_score(query_time, item['lastVisitTime'], 2592000))
                        for item in history_data
                    ]
                    top_results['time'] = sorted(time_results, key=lambda x: x[1], reverse=True)[:10]
                # For specific time queries
                elif rank == -1:
                    dt = datetime.strptime(features['time']['start_date'], '%Y-%m-%dT%H:%M:%S')
                    query_time = int(dt.timestamp()) * 1000

                    time_results = [
                        (item, calculate_time_score(query_time, item['lastVisitTime'], 2592000))
                        for item in history_data
                    ]
                    top_results['time'] = sorted(time_results, key=lambda x: x[1])[:10]
                # For most recent or second most recent
                elif rank > 0:
                    # Takes all the top N ranks
                    dt = datetime.strptime(features['time']['end_date'], '%Y-%m-%dT%H:%M:%S')
                    query_time = int(dt.timestamp()) * 1000

                    time_results = [
                        (item, calculate_time_score(query_time, item['lastVisitTime'], 2592000))
                        for item in history_data
                    ]
                    top_results['time'] = sorted(time_results, key=lambda x: x[1], reverse=True)[:rank]
                # For farthest
                else:
                    # Take all the top N the farthest ranks
                    time_results = [
                        (item, calculate_time_score(features['time']['value'], item['lastVisitTime'], 2592000))
                        for item in history_data
                    ]
                    top_results['time'] = sorted(time_results, key=lambda x: x[1])[:abs(rank)]

            if features.get('color'):
                color_results = [
                    (item, calculate_color_score(features['color'], item.get('color', [0, 0, 0])))
                    for item in history_data
                    if 'color' in item
                ]
                top_results['color'] = sorted(color_results, key=lambda x: x[1], reverse=True)[:10]

            if features.get('category'):
                category_results = [
                    (item, calculate_category_score(features['category'], item.get('categories', [])))
                    for item in history_data
                ]
                top_results['category'] = sorted(category_results, key=lambda x: x[1], reverse=True)[:10]

            # Combine all top results
            all_top_items = []
            for feature_results in top_results.values():
                if feature_results:
                    for result in feature_results:
                        if isinstance(result, tuple) and len(result) == 2:
                            item = result[0]
                            # Thêm item nếu chưa có trong danh sách
                            if item not in all_top_items:
                                all_top_items.append(item)

            # Calculate comprehensive scores for all top items
            final_results = []
            for item in all_top_items:
                title_score = next((score for i, score in top_results['title'] if i == item), 0)
                content_score = next((score for i, score in top_results['content'] if i == item), 0)
                time_score = next((score for i, score in top_results['time'] if i == item), 0)
                color_score = next((score for i, score in top_results['color'] if i == item), 0)
                category_score = next((score for i, score in top_results['category'] if i == item), 0)

                total_score = (
                    0.4 * title_score +
                    0.4 * content_score +
                    0.1 * time_score +
                    0.05 * color_score +
                    0.05 * category_score
                )

                final_results.append((item, total_score))

            # Sort the results by total score in descending order
            sorted_results = sorted(final_results, key=lambda x: x[1], reverse=True)

            # Return the top result(s)
            if sorted_results:
                top_results = sorted_results[:1]  # Adjust this if you want to return more than one result
                response = []
                for match, score in top_results:
                    logging.info("Match: %s, score: %f", match['title'], score)
                    response.append({
                        'id': match.get('id'),
                        'url': match['url'],
                        'title': match['title'],
                        'lastVisitTime': match['lastVisitTime'],
                        'score': score
                    })
                print('Response: ', response)
                return jsonify({'response': response})
            else:
                return jsonify({'response': "Không tìm thấy kết quả phù hợp."})
        else:
            return jsonify({'response': "Không thể trích xuất đặc trưng từ câu hỏi."})
    else:
        return jsonify({'response': "Câu hỏi này không liên quan đến lịch sử duyệt web."})

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


@app.route('/upload_history', methods=['POST'])
def upload_history():
    global history_data, title_vectorstore, content_vectorstore, history_buffer

    new_history_data = request.json.get('history', [])
    if not new_history_data:
        return jsonify({'status': 'error', 'message': 'No history data provided'}), 400

    # Lọc các bản ghi trùng lặp
    new_history_data = filter_duplicate_history(new_history_data)
    logging.info("Received %d new history items", len(new_history_data))

    # Xử lý tất cả các bản ghi trong new_history_data
    processed_items = process_history_items(new_history_data)

    # Thêm các bản ghi đã xử lý vào dữ liệu lịch sử chính
    history_data.extend(processed_items)

    # Tái phân loại các line lịch sử
    history_lines = group_history_into_lines(history_data)

    logging.info("Processed %d history items into %d lines", len(history_data), len(history_lines))
    return jsonify({'status': 'success', 'history_data': history_lines})


def process_history_items(history_items):
    global history_data, title_vectorstore, content_vectorstore
    processed_items = []
    for item in history_items:
        if item.get('is_embedded', True):
            continue

        # Crawl content from URL
        content, sentences = crawl_website_content(item['url'])
        if not content:
            item['content'] = []
            logging.warning("Failed to crawl content from URL: %s", item['url'])
        else:
            item['content'] = sentences

        # Get dominant color from URL
        color = get_dominant_color(item['url'])
        if not color:
            logging.warning("Failed to get dominant color from URL: %s", item['url'])
        else:
            item['color'] = color

        # Classify website based on content and title
        categories = classify_website(item['title'], content)
        item['categories'] = categories

        # Create embeddings for title and content
        title_embedding = get_normalized_embeddings([item['title']])
        content_embedding = get_normalized_embeddings([" ".join(item['content'])])

        # Add embeddings to vectorstore
        if title_vectorstore is None:
            title_vectorstore = faiss.IndexFlatIP(title_embedding.shape[1])
        title_vectorstore.add(title_embedding)

        if content_vectorstore is None:
            content_vectorstore = faiss.IndexFlatIP(content_embedding.shape[1])
        content_vectorstore.add(content_embedding)

        # Mark this item as embedded
        item['is_embedded'] = True

        processed_items.append(item)
    return processed_items


def group_history_into_lines(history_data, time_threshold=7200, interruption_threshold=300):
    lines = []
    current_line = []
    buffer = []  # Lưu tạm các bản ghi không thể gán ngay

    # Sắp xếp lịch sử theo thời gian
    sorted_history = sorted(history_data, key=lambda x: x['lastVisitTime'])

    while sorted_history or buffer:
        if not sorted_history and buffer:
            # Khi danh sách chính rỗng nhưng buffer còn dữ liệu
            remaining_buffer = buffer[:]  # Sao chép buffer để xử lý từng phần tử
            buffer.clear()

            for item in remaining_buffer:
                # Kiểm tra xem có thể nối vào current_line không
                if current_line:
                    last_item = current_line[-1]
                    time_diff = (item['lastVisitTime'] - last_item['lastVisitTime']) / 1000
                    same_domain = are_urls_related(item['url'], last_item['url'])
                    same_category = set(item.get('categories', [])) & set(last_item.get('categories', []))

                    if time_diff <= time_threshold and (same_domain or same_category):
                        current_line.append(item)
                        continue  # Chuyển sang phần tử tiếp theo

                # Nếu không nối được, tạo một line mới
                if current_line:
                    lines.append(current_line)
                    current_line = []

                # Tạo line mới cho item hiện tại
                current_line.append(item)

            # Nếu current_line còn phần tử, thêm vào lines
            if current_line:
                lines.append(current_line)
                current_line = []

            continue  # Quay lại vòng lặp chính

        # Xử lý phần tử trong sorted_history như bình thường
        item = sorted_history.pop(0)

        if not current_line:
            # Nếu current_line rỗng, khởi tạo line mới
            current_line.append(item)
        else:
            last_item = current_line[-1]
            time_diff = (item['lastVisitTime'] - last_item['lastVisitTime']) / 1000
            same_domain = are_urls_related(item['url'], last_item['url'])
            same_category = set(item.get('categories', [])) & set(last_item.get('categories', []))

            if time_diff <= time_threshold and (same_domain or same_category):
                # Nếu thuộc cùng line
                current_line.append(item)
            else:
                # Nếu không thuộc cùng line, kết thúc current_line và bắt đầu line mới
                lines.append(current_line)
                current_line = [item]

    # Gộp line hiện tại (nếu có) vào danh sách lines
    if current_line:
        lines.append(current_line)

    # Thêm thông tin line_id, prev_line, next_line
    for line in lines:
        for i, item in enumerate(line):
            item['line_id'] = f"line_{lines.index(line)}"
            item['prev_item'] = line[i - 1]['id'] if i > 0 else None
            item['next_item'] = line[i + 1]['id'] if i < len(line) - 1 else None

    return lines


def reassign_buffer_to_line(buffer, current_line, time_threshold):
    """
    Kiểm tra từng mục trong buffer và gán vào line hiện tại nếu có thể.
    """
    reassigned_items = []
    for item in buffer:
        last_item = current_line[-1]
        time_diff = (item['lastVisitTime'] - last_item['lastVisitTime']) / 1000
        same_domain = are_urls_related(item['url'], last_item['url'])
        same_category = set(item.get('categories', [])) & set(last_item.get('categories', []))

        if time_diff <= time_threshold and (same_domain or same_category):
            reassigned_items.append(item)

    # Xóa các mục đã gán vào line khỏi buffer
    for item in reassigned_items:
        buffer.remove(item)

    return reassigned_items


def are_urls_related(url1, url2):
    parsed1 = urlparse(url1)
    parsed2 = urlparse(url2)

    # Check if domains are the same or subdomains of each other
    domain1 = parsed1.netloc.split('.')
    domain2 = parsed2.netloc.split('.')

    if domain1[-2:] == domain2[-2:]:  # Same top-level and second-level domain
        return True

    # Check for common paths
    path1 = parsed1.path.strip('/').split('/')
    path2 = parsed2.path.strip('/').split('/')

    if len(path1) > 0 and len(path2) > 0 and path1[0] == path2[0]:
        return True

    return False


def can_reconnect_to_line(item, line, time_threshold):
    for line_item in reversed(line):
        time_diff = (item['lastVisitTime'] - line_item['lastVisitTime']) / 1000
        if time_diff <= time_threshold and (are_urls_related(item['url'], line_item['url']) or
                                            set(item.get('categories', [])) & set(line_item.get('categories', []))):
            return True
    return False


def merge_interruptions(line, interruption_buffer):
    if interruption_buffer:
        # Sort interruptions by time and insert them into the correct position in the line
        sorted_interruptions = sorted(interruption_buffer, key=lambda x: x['lastVisitTime'])
        for interruption in sorted_interruptions:
            insert_position = next(
                (i for i, item in enumerate(line) if item['lastVisitTime'] > interruption['lastVisitTime']), len(line))
            line.insert(insert_position, interruption)
        interruption_buffer.clear()


@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    transcription = transcribe_audio(audio_file)
    return jsonify({"transcription": transcription})


if __name__ == '__main__':
    logging.info("Starting server on port 5000")
    server = pywsgi.WSGIServer(('localhost', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
