browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'getPageDetails') {
        fetch(message.url)
            .then(response => response.text())
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');

                const title = doc.querySelector('title') ? doc.querySelector('title').innerText : 'No title found';
                const content = extractMainContent(doc);
                const colors = extractColorsFromPage(doc);

                sendResponse({ title, content, colors });
            })
            .catch(error => {
                console.error('Error fetching the page:', error);
                sendResponse({ error: error.message });
            });
        return true;
    }
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

browser.webNavigation.onCompleted.addListener((details) => {
    if (details.frameId === 0) { // Chỉ xử lý khi toàn bộ trang web đã tải xong (không phải iframe)
        const url = details.url;
        const visitTime = Date.now();

        saveHistoryToDB(url, visitTime);

        // Tải xuống source của trang web và phân tích nội dung
        fetchPageSourceAndAnalyze(url).then(analysisResult => {
            // Lưu kết quả phân tích vào IndexedDB
            saveAnalysisToDB(url, analysisResult);
        }).catch(error => console.error('Failed to analyze page:', error));

        // Gửi thông điệp tới `history.js` để cập nhật danh sách lịch sử
        browser.runtime.sendMessage({
            action: 'newHistoryItem',
            item: { url, visitTime }
        });
    }
});

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

    browser.history.search({
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
                await storeHistoryItem(pageData);
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
