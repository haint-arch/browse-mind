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

chrome.webNavigation.onCompleted.addListener((details) => {
    if (details.frameId === 0) { // Chỉ xử lý khi toàn bộ trang web đã tải xong (không phải iframe)
        const url = details.url;
        const visitTime = Date.now();

        saveHistoryToDB(url, visitTime);

        console.log('Page loaded:', url, isSearchURL(url));

        if (!isSearchURL(url)) {
            saveHistoryToDB(url, visitTime);

            // Tải xuống source của trang web và phân tích nội dung
            fetchPageSourceAndAnalyze(url).then(analysisResult => {
                // Lưu kết quả phân tích vào IndexedDB
                saveAnalysisToDB(url, analysisResult);
            }).catch(error => console.error('Failed to analyze page:', error));

            // Gửi thông điệp tới `history.js` để cập nhật danh sách lịch sử
            chrome.runtime.sendMessage({
                action: 'newHistoryItem',
                item: { url, visitTime }
            });
        }
    }
});

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

function fetchPageSourceAndAnalyze(url) {
    return fetch(url)
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const title = doc.querySelector('title') ? doc.querySelector('title').innerText : 'No title found';
            const content = extractMainContent(doc);
            const colors = extractColorsFromPage(doc);

            return {
                title,
                content,
                colors
            };
        });
}

function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('historyDB', 1);

        request.onupgradeneeded = event => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('history')) {
                db.createObjectStore('history', { keyPath: 'id', autoIncrement: true });
            }
        };

        request.onsuccess = event => {
            resolve(event.target.result);
        };

        request.onerror = event => {
            reject(event.target.error);
        };
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
    }
});

async function storeHistoryItem(pageData) {
    const db = await openDatabase();
    const tx = db.transaction('history', 'readwrite');
    const store = tx.objectStore('history');

    return new Promise((resolve, reject) => {
        const request = store.add(pageData);
        request.onsuccess = () => resolve();
        request.onerror = (event) => reject(event.target.error);
    });
}

// Hàm để lấy lịch sử trình duyệt trong một tháng qua và lưu vào DB
async function fetchAndStoreHistory() {
    const oneMonthAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);

    chrome.history.search({
        text: '',
        startTime: oneMonthAgo,
        maxResults: 1000      // Giới hạn số lượng kết quả 
    }).then(async (results) => {
        for (const item of results) {
            const pageData = {
                url: item.url,
                title: item.title,
                lastVisitTime: item.lastVisitTime
            };

            try {
                if (!isSearchURL(pageData.url)) {
                    // const analysisResult = await fetchPageSourceAndAnalyze(pageData.url);
                    // pageData.analysis = analysisResult;
                    await storeHistoryItem(pageData);
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
