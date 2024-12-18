document.addEventListener('DOMContentLoaded', () => {
    let recorder;
    let isRecording = false;
    let startTime;
    let elapsedTimeInterval;
    let audioChunks = [];

    const speechToTextLoading = document.getElementById('speechToTextLoading');
    const chatInput = document.getElementById('chatInput');
    const chatButton = document.getElementById('chatButton');
    const chatResponse = document.getElementById('chatResponse');
    const assistantHeader = document.getElementById('assistantHeader');
    const headerH2 = assistantHeader.querySelector('h2');
    const recordButton = document.getElementById('recordButton');
    const micIcon = document.getElementById('micIcon');
    const settingsButton = document.getElementById('settingsButton');
    const settingsDropdown = document.getElementById('settingsDropdown');
    const softwareInfoItem = document.getElementById('softwareInfo');
    const manageHistoryItem = document.getElementById('manageHistory');

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

    function toggleDropdown(show) {
        settingsButton.classList.toggle('active', show);
        settingsDropdown.classList.toggle('show', show);
    }

    settingsButton.addEventListener('click', (event) => {
        event.stopPropagation();
        const isShowing = settingsDropdown.classList.contains('show');
        toggleDropdown(!isShowing);
    });

    document.addEventListener('click', (event) => {
        if (!settingsButton.contains(event.target) && !settingsDropdown.contains(event.target)) {
            toggleDropdown(false);
        }
    });

    // Add keyboard navigation
    settingsDropdown.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            toggleDropdown(false);
            settingsButton.focus();
        }
    });

    function handleMenuItemClick(action) {
        return () => {
            toggleDropdown(false);
            if (action === 'Software Info') {
                chrome.tabs.create({ url: 'introduction.html' });
            } else if (action === 'Manage History') {
                chrome.tabs.create({ url: 'history-management.html' });
            }
        };
    }

    softwareInfoItem.addEventListener('click', handleMenuItemClick('Software Info'));
    manageHistoryItem.addEventListener('click', handleMenuItemClick('Manage History'));

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
            showLoadingAnimation(true, 'Đang tải lịch sử lên máy chủ...');

            let allItems = await fetchHistoryFromIndexedDB();
            const jsonData = JSON.stringify(allItems, null, 2);
            console.log("Dữ liệu dạng JSON:", jsonData);
            

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
            chrome.runtime.sendMessage(
                {
                    action: 'updateProcessedItems',
                    historyData: historyData
                },
                (response) => {
                    if (response.error) {
                        console.error('Error updating processed items:', response.error);
                        reject(response.error);
                    } else {
                        console.log('Successfully updated processed items with line information');
                        resolve();
                    }
                }
            );
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

    // loadRecentChat();
    initializeHistory();

    // function loadRecentChat() {
    //     const recentChat = JSON.parse(localStorage.getItem('recentChat')) || {};
    //     if (recentChat.question) chatInput.value = recentChat.question;
    //     if (recentChat.response && Array.isArray(recentChat.response)) {
    //         for (const item of recentChat.response) {
    //             console.log('Recent chat item:', typeof item, item);
                
    //             const fullHistoryItems = { ...item };
    //             const responseElement = createResponseElements([...fullHistoryItems]);       
    //             chatResponse.appendChild(responseElement);
    //         }
    //     }
    //     if (recentChat.question || (recentChat.response && recentChat.response.length > 0)) {
    //         chatInput.value = '';
    //         assistantHeader.style.height = '';
    //         if (headerH2) headerH2.remove();
    //     }
    // }

    function saveRecentChat(question, response) {
        localStorage.setItem('recentChat', JSON.stringify({ question, response }));
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
                const fullHistoryItems = await Promise.all(data.response.map(async item => {
                    const fullItem = await getFullHistoryItem(item.id);
                    return { ...item, ...fullItem };
                }));
                for (const item of fullHistoryItems) {
                    console.log('History item:', typeof item, item);
                    const responseElement = await createResponseElements([item]);
                    chatResponse.appendChild(responseElement);
                }
                saveRecentChat(message, fullHistoryItems);
            } else {
                throw new Error('Unexpected response format');
            }
    
            chatInput.value = '';
        } catch (error) {
            console.error('Error communicating with the chatbot:', error);
            const errorElement = document.createElement('div');
            errorElement.textContent = 'Error communicating with the chatbot.';
            errorElement.classList.add('response-message');
            chatResponse.appendChild(errorElement);
    
            saveRecentChat(message, 'Error communicating with the chatbot.');
        } finally {
            const loadingElement = assistantHeader.querySelector('.loading');
            if (loadingElement) {
                assistantHeader.removeChild(loadingElement);
            }
            assistantHeader.style.height = '';
        }
    }

    async function getFullHistoryItem(idOrUrl) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage(
                { action: 'getHistoryItem', idOrUrl: idOrUrl },
                (response) => {
                    if (response.error) {
                        console.error('Error fetching history item:', response.error);
                        reject(response.error);
                    } else {
                        resolve(response.item);
                    }
                }
            );
        });
    }

    async function createResponseElements(historyItems) {
        const containerElement = document.createElement('div');
        containerElement.classList.add('responses-container');
    
        for (const item of historyItems) {
            const historyContainer = await createHistoryContainer(item);
            containerElement.appendChild(historyContainer);
        }
    
        return containerElement;
    }

    async function createHistoryContainer(item) {
        const historyContainer = document.createElement('div');
        historyContainer.classList.add('history-container');
        historyContainer.dataset.lineId = item.line_id;
    
        // Create navigation buttons
        const navigationElement = document.createElement('div');
        navigationElement.classList.add('history-navigation');
    
        const prevButton = document.createElement('button');
        prevButton.innerHTML = '&lt;';
        prevButton.disabled = !item.prev_item;
        prevButton.onclick = () => navigateToItem(item.prev_item);
    
        const nextButton = document.createElement('button');
        nextButton.innerHTML = '&gt;';
        nextButton.disabled = !item.next_item;
        nextButton.onclick = () => navigateToItem(item.next_item);
    
        navigationElement.appendChild(prevButton);
        navigationElement.appendChild(nextButton);
        historyContainer.appendChild(navigationElement);
    
        // Create item content
        const itemElement = createItemElement(item);
        historyContainer.appendChild(itemElement);
    
        // Create containers for previous and next items
        // if (item.prev_item) {
        //     const prevItem = await getFullHistoryItem(item.prev_item);
        //     const prevItemContainer = document.createElement('div');
        //     prevItemContainer.classList.add('adjacent-item', 'prev-item');
        //     prevItemContainer.appendChild(createItemElement(prevItem));
        //     historyContainer.appendChild(prevItemContainer);
        // }
    
        // if (item.next_item) {
        //     const nextItem = await getFullHistoryItem(item.next_item);
        //     const nextItemContainer = document.createElement('div');
        //     nextItemContainer.classList.add('adjacent-item', 'next-item');
        //     nextItemContainer.appendChild(createItemElement(nextItem));
        //     historyContainer.appendChild(nextItemContainer);
        // }
    
        return historyContainer;
    }

    function createItemElement(item) {
        const itemElement = document.createElement('div');
        itemElement.classList.add('history-item');

        const titleElement = document.createElement('a');
        titleElement.href = item.url;
        titleElement.textContent = item.title;
        titleElement.target = '_blank';
        itemElement.appendChild(titleElement);

        // if (item.score !== undefined) {
        //     const scoreElement = document.createElement('div');
        //     scoreElement.classList.add('similarity');
        //     scoreElement.textContent = `Score: ${item.score.toFixed(4)}`;
        //     itemElement.appendChild(scoreElement);
        // }

        if (item.categories && item.categories.length > 0) {
            const categoriesElement = document.createElement('div');
            categoriesElement.classList.add('categories');
            categoriesElement.textContent = `Categories: ${item.categories.join(', ')}`;
            itemElement.appendChild(categoriesElement);
        }

        if (item.color) {
            const colorElement = document.createElement('div');
            colorElement.classList.add('color');
            colorElement.textContent = `Color: rgb(${item.color.join(', ')})`;
            colorElement.style.backgroundColor = `rgb(${item.color.join(', ')})`;
            colorElement.style.color = getContrastColor(item.color);
            itemElement.appendChild(colorElement);
        }

        const timeElement = document.createElement('div');
        timeElement.classList.add('time');
        timeElement.textContent = `Last Visited: ${new Date(item.lastVisitTime).toLocaleString()}`;
        itemElement.appendChild(timeElement);

        return itemElement;
    }

    function getContrastColor(rgb) {
        const brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000;
        return brightness > 128 ? 'black' : 'white';
    }

    async function navigateToItem(itemId) {
        if (!itemId) return;
    
        const item = await getFullHistoryItem(itemId);
        if (!item) return;
    
        const containers = document.querySelectorAll('.history-container');
        containers.forEach(container => {
            if (container.dataset.lineId === item.line_id) {
                container.style.display = 'block';
                updateHistoryContainer(container, item);
            } else {
                container.style.display = 'none';
            }
        });
    }

    async function updateHistoryContainer(container, item) {
        // Update main item
        const itemElement = container.querySelector('.history-item');
        itemElement.innerHTML = '';
        itemElement.appendChild(createItemElement(item));
    
        // Update navigation buttons
        const prevButton = container.querySelector('.history-navigation button:first-child');
        const nextButton = container.querySelector('.history-navigation button:last-child');
    
        prevButton.disabled = !item.prev_item;
        nextButton.disabled = !item.next_item;
    
        prevButton.onclick = () => navigateToItem(item.prev_item);
        nextButton.onclick = () => navigateToItem(item.next_item);
    
        // Update previous item container
        // let prevItemContainer = container.querySelector('.prev-item');
        // if (item.prev_item) {
        //     const prevItem = await getFullHistoryItem(item.prev_item);
        //     if (!prevItemContainer) {
        //         prevItemContainer = document.createElement('div');
        //         prevItemContainer.classList.add('adjacent-item', 'prev-item');
        //         container.appendChild(prevItemContainer);
        //     }
        //     prevItemContainer.innerHTML = '';
        //     prevItemContainer.appendChild(createItemElement(prevItem));
        // } else if (prevItemContainer) {
        //     prevItemContainer.remove();
        // }
    
        // Update next item container
        // let nextItemContainer = container.querySelector('.next-item');
        // if (item.next_item) {
        //     const nextItem = await getFullHistoryItem(item.next_item);
        //     if (!nextItemContainer) {
        //         nextItemContainer = document.createElement('div');
        //         nextItemContainer.classList.add('adjacent-item', 'next-item');
        //         container.appendChild(nextItemContainer);
        //     }
        //     nextItemContainer.innerHTML = '';
        //     nextItemContainer.appendChild(createItemElement(nextItem));
        // } else if (nextItemContainer) {
        //     nextItemContainer.remove();
        // }
    }

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
                chatInput.style.display = 'none';
                speechToTextLoading.style.display = 'flex';

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
            } finally {
                speechToTextLoading.style.display = 'none';
                chatInput.style.display = 'block';
                isRecording = false;
                updateMicIcon(false);
            }
        };
    }

    function updateMicIcon(isRecording) {
        recordButton.classList.toggle('recording', isRecording);
    }

    function startTimer() {
        startTime = Date.now();
        elapsedTimeInterval = setInterval(() => {
            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
            chatResponse.innerHTML = `Recording (${elapsedTime}s)`;
        }, 100);
    }

    function stopTimer() {
        clearInterval(elapsedTimeInterval);
        chatResponse.innerHTML = '';
    }

    window.navigateToLine = navigateToItem;
});