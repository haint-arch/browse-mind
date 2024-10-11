document.addEventListener('DOMContentLoaded', () => {
    let recorder;
    let audioChunks = [];
    let isRecording = false;
    let startTime;
    let elapsedTimeInterval;

    const chatInput = document.getElementById('chatInput');
    const chatButton = document.getElementById('chatButton');
    const chatResponse = document.getElementById('chatResponse');
    // const historyButton = document.getElementById('historyButton');
    const assistantHeader = document.getElementById('assistantHeader');

    const headerH2 = assistantHeader.querySelector('h2');

    const recordButton = document.getElementById('recordButton');
    const micIcon = document.getElementById('micIcon');

    micIcon.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // Function to load the most recent chat from localStorage
    const loadRecentChat = () => {
        const recentChat = JSON.parse(localStorage.getItem('recentChat')) || {};
        if (recentChat.question) {
            chatInput.value = recentChat.question;
        }
        if (recentChat.response) {
            const messageElement = createResponseElement(recentChat.response);
            chatResponse.appendChild(messageElement);
        }
        // If recentChat exists, clear chatInput and shrink assistantHeader
        if (recentChat.question || recentChat.response) {
            chatInput.value = '';
            assistantHeader.style.height = '';
            headerH2.remove();
        }
    };

    // Function to save the most recent chat to localStorage
    const saveRecentChat = (question, response) => {
        const recentChat = { question, response };
        localStorage.setItem('recentChat', JSON.stringify(recentChat));
    };

    // Function to create response element
    const createResponseElement = (responseText) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('response-message');

        // Extract title, URL, and similarity from response
        const regex = /Câu tiêu đề liên quan nhất: (.+) \(Độ tương đồng: (.+)\)\. URL: (.+)/;
        const match = responseText.match(regex);

        if (match) {
            const title = match[1];
            const similarity = match[2];
            const url = match[3];

            // Create title element as <a>
            const titleElement = document.createElement('a');
            titleElement.href = url;
            titleElement.textContent = title;
            titleElement.target = '_blank';
            messageElement.appendChild(titleElement);

            // Create similarity element
            const similarityElement = document.createElement('div');
            similarityElement.textContent = `Độ tương đồng: ${similarity}`;
            similarityElement.classList.add('similarity');
            messageElement.appendChild(similarityElement);
        } else {
            messageElement.textContent = responseText;
        }

        return messageElement;
    };

    const sendMessage = async () => {
        const message = chatInput.value.trim();
        if (message) {
            try {
                // Set h2 innerText to '' and adjust container height smoothly
                if (headerH2) {
                    assistantHeader.style.height = '100px';  // Shrink header
                    headerH2.remove();
                }

                // Add loading spinner
                const loadingElement = document.createElement('div');
                loadingElement.classList.add('loading');
                assistantHeader.appendChild(loadingElement);

                // Clear previous responses
                chatResponse.innerHTML = '';

                // Fetch the response
                const response = await fetch('http://localhost:5000/chatbot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: message })
                });

                const data = await response.json();
                const messageElement = createResponseElement(data.response);
                chatResponse.appendChild(messageElement);

                // Save response to localStorage
                saveRecentChat(message, data.response);

                // Check if recentChat exists and perform actions
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

                // Save error message to localStorage
                saveRecentChat(message, 'Error communicating with the chatbot.');

                if (loadingElement && assistantHeader.contains(loadingElement)) {
                    assistantHeader.removeChild(loadingElement);
                    assistantHeader.style.height = '';
                }
            }
        }
    };

    async function startRecording() {
        try {
            // Yêu cầu quyền truy cập microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Tạo MediaRecorder và ghi âm
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = (e) => {
                audioChunks.push(e.data);
            };

            recorder.onstart = () => {
                startTime = Date.now();
                updateMicIcon(true);  // Cập nhật nút micro thành màu đỏ
                startTimer();  // Hiển thị thời gian ghi âm
            };

            recorder.start();
            isRecording = true;
        } catch (err) {
            console.error('Lỗi khi truy cập microphone:', err);
            alert('Vui lòng cấp quyền truy cập microphone.');
        }
    }

    // Dừng ghi âm và gửi file .wav đến API
    async function stopRecording() {
        recorder.stop();
        recorder.onstop = async () => {
            updateMicIcon(false);  // Cập nhật nút micro thành màu trắng
            stopTimer();  // Dừng hiển thị thời gian ghi âm

            // Tạo Blob từ dữ liệu âm thanh đã ghi
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];  // Xóa dữ liệu cũ

            // Gửi file âm thanh đến API /transcribe
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            try {
                const response = await fetch('http://localhost:5000/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                chatResponse.innerHTML = ``;
                chatInput.value = data.transcription;
            } catch (error) {
                console.error('Error sending audio to the server:', error);
                chatResponse.innerHTML = 'Lỗi trong quá trình xử lý âm thanh.';
            }

            isRecording = false;
        };
    }

    // Cập nhật trạng thái nút microphone
    function updateMicIcon(isRecording) {
        if (isRecording) {
            recordButton.classList.add('recording'); 
        } else {
            recordButton.classList.remove('recording'); 
        }
    }

    // Hiển thị thời gian ghi âm
    function startTimer() {
        elapsedTimeInterval = setInterval(() => {
            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
            chatResponse.innerHTML = `Recording (${elapsedTime}s)`;
        }, 100);
    }

    // Dừng hiển thị thời gian ghi âm
    function stopTimer() {
        clearInterval(elapsedTimeInterval);
        chatResponse.innerHTML = 'Transcribing...';
    }

    chatButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    loadRecentChat();

    // historyButton.addEventListener('click', () => {
    //     window.open('history.html', '_blank');
    // });
});
