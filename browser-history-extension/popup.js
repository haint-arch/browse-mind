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

    async function sendHistoryToServerWithLoading() {
        try {
            setChatboxReadonly(true);
            showLoadingAnimation(true, 'Embedding...');

            let historyItems = await fetchHistoryFromIndexedDB();
            historyItems = filterDuplicateHistory(historyItems);

            const response = await fetch('http://localhost:5000/upload_history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history: historyItems })
            });

            if (!response.ok) throw new Error('Failed to upload history to server');
            console.log('History uploaded successfully');
        } catch (error) {
            console.error('Error uploading history:', error);
        } finally {
            setChatboxReadonly(false);
            showLoadingAnimation(false);
        }
    }

    function filterDuplicateHistory(historyItems) {
        const uniqueHistory = [];
        const seenUrls = new Set();

        for (const item of historyItems) {
            if (!seenUrls.has(item.url)) {
                uniqueHistory.push(item);
                seenUrls.add(item.url);
            }
        }
        return uniqueHistory;
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
            await sendHistoryToServerWithLoading();
            await clearIndexedDB();
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
    
    // Call initializeHistory to upload and embed history when the popup is opened
    if (!localStorage.getItem('historyInitialized')) {
        initializeHistory();
        localStorage.setItem('historyInitialized', 'true');
    }

    function loadRecentChat() {
        const recentChat = JSON.parse(localStorage.getItem('recentChat')) || {};
        if (recentChat.question) chatInput.value = recentChat.question;
        if (recentChat.response) {
            const messageElement = createResponseElement(recentChat.response);
            chatResponse.appendChild(messageElement);
        }
        if (recentChat.question || recentChat.response) {
            chatInput.value = '';
            assistantHeader.style.height = '';
            headerH2.remove();
        }
    }

    function saveRecentChat(question, response) {
        localStorage.setItem('recentChat', JSON.stringify({ question, response }));
    }

    function createResponseElement(responseText) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('response-message');

        const regex = /Câu tiêu đề liên quan nhất: (.+) \(Độ tương đồng: (.+)\)\. URL: (.+)/;
        const match = responseText.match(regex);

        if (match) {
            const [_, title, similarity, url] = match;

            const titleElement = document.createElement('a');
            titleElement.href = url;
            titleElement.textContent = title;
            titleElement.target = '_blank';
            messageElement.appendChild(titleElement);

            const similarityElement = document.createElement('div');
            similarityElement.textContent = `Độ tương đồng: ${similarity}`;
            similarityElement.classList.add('similarity');
            messageElement.appendChild(similarityElement);
        } else {
            messageElement.textContent = responseText;
        }

        return messageElement;
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
            const messageElement = createResponseElement(data.response);
            chatResponse.appendChild(messageElement);

            saveRecentChat(message, data.response);

            const recentChat = JSON.parse(localStorage.getItem('recentChat'));
            if (recentChat) {
                assistantHeader.removeChild(loadingElement);
                assistantHeader.style.height = '';
                chatInput.value = '';
            }
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