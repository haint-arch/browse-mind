if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('background.js')
        .then(function (registration) {
            console.log('Service Worker registered with scope:', registration.scope);
        })
        .catch(function (error) {
            console.error('Service Worker registration failed:', error);
        });
}

chrome.runtime.onInstalled.addListener(() => {
    console.log("Service Worker đã được cài đặt!");
});

function extractMainContent(doc) {
    const mainContent = doc.querySelector('article') || doc.querySelector('main') || doc.body;
    return mainContent ? mainContent.innerText.trim().substring(0, 500) : 'No content found';
}

function extractColorsFromPage(doc) {
    const styles = Array.from(doc.querySelectorAll('*'))
        .map(el => window.getComputedStyle(el).backgroundColor)
        .filter(color => color && color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent');

    return [...new Set(styles)];
}

chrome.webNavigation.onCompleted.addListener(async (details) => {
    if (details.frameId === 0) { // Chỉ xử lý khi toàn bộ trang web đã tải xong (không phải iframe)
        const url = details.url;
        const visitTime = Date.now();

        // Lấy tiêu đề của trang web
        const title = await fetchPageTitle(url);

        // Lưu vào IndexedDB
        const pageData = {
            url: url,
            title: title,
            lastVisitTime: visitTime,
            is_embedded: false
        };
        const id = await storeHistoryItem(pageData);
        pageData.id = id;

        console.log('Page loaded:', pageData);

        if (!isSearchURL(url)) {
            // Gửi bản ghi mới về backend API /upload_history
            uploadHistory([pageData]);
        }
    }
});

async function fetchPageTitle(url) {
    try {
        const response = await fetch(url);
        const html = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        return doc.querySelector('title') ? doc.querySelector('title').innerText : 'No title found';
    } catch (error) {
        console.error('Failed to fetch page title:', error);
        return 'No title found';
    }
}

async function uploadHistory(newHistoryData) {
    try {
        const response = await fetch('http://localhost:5000/upload_history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ history: newHistoryData })
        });

        if (!response.ok) throw new Error('Failed to upload history to server');
        const { history_data } = await response.json();
        console.log('Processed history data:', history_data);

        // Update IndexedDB with processed items
        await updateProcessedItems(history_data);
    } catch (error) {
        console.error('Error uploading history:', error);
    }
}

function isSearchURL(url) {
    const searchEngines = [
        'https://www.google.com/search',
        'https://search.yahoo.com/search',
        'https://www.bing.com/search',
        'https://duckduckgo.com/',
        'https://www.baidu.com/s',
        'https://yandex.com/search/',
        'https://www.ask.com/web',
        'https://search.aol.com/aol/search',
        'https://www.ecosia.org/search'
    ];

    return searchEngines.some(engine => url.startsWith(engine));
}

function saveHistoryToDB(url, visitTime) {
    openDatabase().then(db => {
        const tx = db.transaction('history', 'readwrite');
        const store = tx.objectStore('history');
        store.put({ url, visitTime });
        tx.oncomplete = () => console.log('History item saved:', url);
    }).catch(error => console.error('Failed to save history item:', error));
}

function saveAnalysisToDB(url, analysisResult) {
    openDatabase().then(db => {
        const tx = db.transaction('history', 'readwrite');
        const store = tx.objectStore('history');
        const request = store.get(url);

        request.onsuccess = () => {
            const record = request.result;
            if (record) {
                record.analysis = analysisResult;
                store.put(record);
            } else {
                store.put({ url, analysis: analysisResult });
            }
            tx.oncomplete = () => console.log('Page analysis saved for:', url);
        };
    }).catch(error => console.error('Failed to save page analysis:', error));
}

function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('historyDB', 2);

        request.onupgradeneeded = event => {
            const db = event.target.result;
            const store = db.objectStoreNames.contains('history')
                ? event.target.transaction.objectStore('history')
                : db.createObjectStore('history', { keyPath: 'id', autoIncrement: true });

            if (!store.indexNames.contains('is_embedded')) {
                store.createIndex('is_embedded', 'is_embedded', { unique: false });
            }
        };

        request.onsuccess = event => resolve(event.target.result);
        request.onerror = event => reject(event.target.error);
    });
}

function getAllHistoryItems() {
    return new Promise((resolve, reject) => {
        openDatabase().then(db => {
            const tx = db.transaction('history', 'readonly');
            const store = tx.objectStore('history');
            const request = store.getAll();

            request.onsuccess = () => {
                resolve(request.result);
            };

            request.onerror = (event) => {
                reject(event.target.error);
            };
        }).catch(error => reject(error));
    });
}

function updateProcessedItems(historyData) {
    return new Promise((resolve, reject) => {
        openDatabase().then(db => {
            const tx = db.transaction('history', 'readwrite');
            const store = tx.objectStore('history');

            historyData.forEach(item => {
                const request = store.get(item.id);
                request.onsuccess = () => {
                    const record = request.result;
                    if (record) {
                        record.content = item.content;
                        record.color = item.color;
                        record.categories = item.categories;
                        record.is_embedded = true;
                        store.put(record);
                    }
                };
            });

            tx.oncomplete = () => resolve();
            tx.onerror = (event) => reject(event.target.error);
        }).catch(error => reject(error));
    });
}

function clearHistory() {
    return new Promise((resolve, reject) => {
        openDatabase().then(db => {
            const tx = db.transaction('history', 'readwrite');
            const store = tx.objectStore('history');
            const request = store.clear();

            request.onsuccess = () => {
                console.log('History cleared');
                resolve();
            };

            request.onerror = (event) => {
                reject(event.target.error);
            };
        }).catch(error => reject(error));
    });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'getAllHistoryItems') {
        getAllHistoryItems().then(items => {
            sendResponse({ items });
        }).catch(error => {
            console.error('Error fetching history items:', error);
            sendResponse({ error: error.message });
        });
        return true; // Giữ kênh message mở để gửi phản hồi không đồng bộ
    } else if (message.action === 'clearHistory') {
        clearHistory().then(() => {
            sendResponse({ status: 'success' });
        }).catch(error => {
            console.error('Error clearing history:', error);
            sendResponse({ error: error.message });
        });
        return true; // Giữ kênh message mở để gửi phản hồi không đồng bộ
    } else if (message.action === 'updateProcessedItems') {
        updateProcessedItems(message.historyData).then(() => {
            sendResponse({ status: 'success' });
        }).catch(error => {
            console.error('Error updating processed items:', error);
            sendResponse({ error: error.message });
        });
        return true; // Giữ kênh message mở để gửi phản hồi không đồng bộ
    }
});

async function storeHistoryItem(pageData) {
    const db = await openDatabase();
    const tx = db.transaction('history', 'readwrite');
    const store = tx.objectStore('history');

    return new Promise((resolve, reject) => {
        const request = store.add(pageData);
        request.onsuccess = (event) => resolve(event.target.result);
        request.onerror = (event) => reject(event.target.error);
    });
}

// Hàm để lấy lịch sử trình duyệt trong một tháng qua và lưu vào DB
async function fetchAndStoreHistory() {
    const oneMonthAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    const processedIds = [];

    chrome.history.search({
        text: '',
        startTime: oneMonthAgo,
        maxResults: 1000      // Giới hạn số lượng kết quả 
    }).then(async (results) => {
        for (const item of results) {
            const pageData = {
                url: item.url,
                title: item.title,
                lastVisitTime: item.lastVisitTime,
                is_embedded: false
            };

            try {
                if (!isSearchURL(pageData.url)) {
                    const id = await storeHistoryItem(pageData);
                    processedIds.push(id);
                }
            } catch (error) {
                console.error('Error storing history item:', error);
            }
        }
        console.log('Browsing history stored successfully.');
    }).catch(error => {
        console.error('Error fetching history:', error);
    });
}

// Gọi hàm khi extension được cài đặt hoặc cập nhật
chrome.runtime.onInstalled.addListener(() => {
    fetchAndStoreHistory();
});