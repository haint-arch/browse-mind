document.addEventListener('DOMContentLoaded', () => {
    let recorder;
    let isRecording = false;
    let startTime;
    let elapsedTimeInterval;
    let socket;

    const chatInput = document.getElementById('chatInput');
    const chatButton = document.getElementById('chatButton');
    const chatResponse = document.getElementById('chatResponse');
    const assistantHeader = document.getElementById('assistantHeader');
    const headerH2 = assistantHeader.querySelector('h2');
    const recordButton = document.getElementById('recordButton');
    const micIcon = document.getElementById('micIcon');

    if (navigator.serviceWorker) {
        navigator.serviceWorker.ready.then(registration => {
            registration.active.postMessage({ action: 'activate' });
        });
    }

    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.action === 'keep_alive') {
            console.log('Received keep_alive message');
            sendResponse({ status: 'alive' });
        }
    });

    micIcon.addEventListener('click', () => {
        isRecording ? stopRecording() : startRecording();
    });

    chatButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function fetchHistoryFromIndexedDB() {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({ action: 'getAllHistoryItems' }, (response) => {
                response.error ? reject(response.error) : resolve(response.items);
            });
        });
    }

    async function uploadHistory() {
        try {
            setChatboxReadonly(true);
            showLoadingAnimation(true, 'Uploading history...');
    
            let allItems = await fetchHistoryFromIndexedDB();
    
            const response = await fetch('http://localhost:5000/upload_history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history: allItems })
            });
    
            if (!response.ok) throw new Error('Failed to upload history to server');
            const { history_data } = await response.json();
            console.log('Processed history data:', history_data);
            
            // Update IndexedDB with processed items
            await updateProcessedItems(history_data);
        } catch (error) {
            console.error('Error uploading history:', error);
        } finally {
            setChatboxReadonly(false);
            showLoadingAnimation(false);
        }
    }

    function updateProcessedItems(historyData) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({ action: 'updateProcessedItems', historyData }, (response) => {
                response.error ? reject(response.error) : resolve();
            });
        });
    }

    function setChatboxReadonly(isReadonly) {
        chatInput.readOnly = isReadonly;
        chatButton.disabled = isReadonly;
    }

    function showLoadingAnimation(show, message = '') {
        let loadingElement = document.querySelector('.loading-animation');
        if (show) {
            if (!loadingElement) {
                loadingElement = document.createElement('div');
                loadingElement.classList.add('loading-animation');
                loadingElement.textContent = message;
                document.body.appendChild(loadingElement);
            }
        } else {
            if (loadingElement) {
                loadingElement.remove();
            }
        }
    }

    async function initializeHistory() {
        try {
            await uploadHistory();
            // await clearIndexedDB();
        } catch (error) {
            console.error('Error initializing history:', error);
        }
    }

    async function clearIndexedDB() {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({ action: 'clearHistory' }, (response) => {
                response.error ? reject(response.error) : resolve();
            });
        });
    }

    loadRecentChat();
    initializeHistory();

    function loadRecentChat() {
        const recentChat = JSON.parse(localStorage.getItem('recentChat')) || {};
        if (recentChat.question) chatInput.value = recentChat.question;
        if (recentChat.response && Array.isArray(recentChat.response)) {
            const messageElement = createResponseElement(recentChat.response);
            chatResponse.appendChild(messageElement);
        }
        if (recentChat.question || (recentChat.response && recentChat.response.length > 0)) {
            chatInput.value = '';
            assistantHeader.style.height = '';
            if (headerH2) headerH2.remove();
        }
    }

    function saveRecentChat(question, response) {
        localStorage.setItem('recentChat', JSON.stringify({ question, response }));
    }

    function createResponseElement(responses) {
        const containerElement = document.createElement('div');
        containerElement.classList.add('responses-container');

        responses.forEach((response, index) => {
            const messageElement = document.createElement('div');
            messageElement.classList.add('response-message');

            const titleElement = document.createElement('a');
            titleElement.href = response.url;
            titleElement.textContent = `${index + 1}. ${response.title}`;
            titleElement.target = '_blank';
            messageElement.appendChild(titleElement);

            const similarityElement = document.createElement('div');
            similarityElement.textContent = `Độ tương đồng: ${response.score.toFixed(4)}`;
            similarityElement.classList.add('similarity');
            messageElement.appendChild(similarityElement);

            containerElement.appendChild(messageElement);
        });

        return containerElement;
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        try {
            if (headerH2) {
                assistantHeader.style.height = '100px';
                headerH2.remove();
            }

            const loadingElement = document.createElement('div');
            loadingElement.classList.add('loading');
            assistantHeader.appendChild(loadingElement);

            chatResponse.innerHTML = '';

            const response = await fetch('http://localhost:5000/chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message })
            });

            const data = await response.json();
            console.log('Chatbot response:', data);

            if (typeof data.response === 'string') {
                const messageElement = document.createElement('div');
                messageElement.classList.add('response-message');
                messageElement.textContent = data.response;
                chatResponse.appendChild(messageElement);
            } else if (Array.isArray(data.response) && data.response.length > 0) {
                data.response.forEach(item => {
                    const messageElement = document.createElement('div');
                    messageElement.classList.add('response-message');

                    const titleElement = document.createElement('a');
                    titleElement.href = item.url;
                    titleElement.textContent = item.title;
                    titleElement.target = '_blank';
                    messageElement.appendChild(titleElement);

                    const scoreElement = document.createElement('div');
                    scoreElement.classList.add('similarity');
                    scoreElement.textContent = `Score: ${item.score.toFixed(2)}`;
                    messageElement.appendChild(scoreElement);

                    const categoriesElement = document.createElement('div');
                    categoriesElement.classList.add('similarity');
                    categoriesElement.textContent = `Categories: ${item.categories.join(', ')}`;
                    messageElement.appendChild(categoriesElement);

                    chatResponse.appendChild(messageElement);
                });

                saveRecentChat(message, data.response);
            } else {
                throw new Error('Unexpected response format');
            }

            assistantHeader.removeChild(loadingElement);
            assistantHeader.style.height = '';
            chatInput.value = '';
        } catch (error) {
            console.error('Error communicating with the chatbot:', error);
            const errorElement = document.createElement('div');
            errorElement.textContent = 'Error communicating with the chatbot.';
            errorElement.classList.add('response-message');
            chatResponse.appendChild(errorElement);

            saveRecentChat(message, 'Error communicating with the chatbot.');

            if (loadingElement && assistantHeader.contains(loadingElement)) {
                assistantHeader.removeChild(loadingElement);
                assistantHeader.style.height = '';
            }
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorder = new MediaRecorder(stream);

            recorder.ondataavailable = (e) => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(e.data);
                }
            };

            recorder.onstart = () => {
                startTime = Date.now();
                updateMicIcon(true);
                startTimer();
                socket = new WebSocket('ws://localhost:5000/transcribe_stream');
                socket.onopen = () => console.log('WebSocket connection established');
                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    chatInput.value += data.transcription + ' ';
                };
                socket.onerror = (error) => console.error('WebSocket error:', error);
                socket.onclose = () => console.log('WebSocket connection closed');
            };

            recorder.start(100);
            isRecording = true;
        } catch (err) {
            console.error('Lỗi khi truy cập microphone:', err);
            alert('Vui lòng cấp quyền truy cập microphone.');
        }
    }

    function stopRecording() {
        recorder.stop();
        updateMicIcon(false);
        stopTimer();
        if (socket) socket.close();
        isRecording = false;
    }

    function updateMicIcon(isRecording) {
        recordButton.classList.toggle('recording', isRecording);
    }

    function startTimer() {
        elapsedTimeInterval = setInterval(() => {
            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
            chatResponse.innerHTML = `Recording (${elapsedTime}s)`;
        }, 100);
    }

    function stopTimer() {
        clearInterval(elapsedTimeInterval);
        chatResponse.innerHTML = '';
    }
});