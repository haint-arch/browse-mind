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

async function deleteHistoryItem(id) {
    try {
        const db = await openDatabase();
        const tx = db.transaction('history', 'readwrite');
        const store = tx.objectStore('history');

        await new Promise((resolve, reject) => {
            const request = store.delete(id);
            request.onsuccess = resolve;
            request.onerror = () => reject(request.error);
        });

        console.log(`History item with id ${id} deleted successfully`);
        return { success: true };
    } catch (error) {
        console.error('Error deleting history item:', error);
        return { success: false, error: error.message };
    }
}

// Function to update a history item
async function updateHistoryItem(item) {
    try {
        const db = await openDatabase();
        const tx = db.transaction('history', 'readwrite');
        const store = tx.objectStore('history');

        await new Promise((resolve, reject) => {
            const request = store.put(item);
            request.onsuccess = resolve;
            request.onerror = () => reject(request.error);
        });

        console.log(`History item with id ${item.id} updated successfully`);
        return { success: true };
    } catch (error) {
        console.error('Error updating history item:', error);
        return { success: false, error: error.message };
    }
}

chrome.webNavigation.onCompleted.addListener(async (details) => {
    if (details.frameId === 0) { // Chỉ xử lý khi toàn bộ trang web đã tải xong (không phải iframe)
        const { tabId, url } = details;
        const visitTime = Date.now();
        // Kiểm tra URL hợp lệ
        if (!url || !isValidURL(url) || isExcludedURL(url) || isSearchURL(url) || isRedirectURL(url)) {
            console.warn('Invalid, excluded, search, or redirect URL:', url);
            return;
        }

        // Lấy tiêu đề của trang web
        const tab = await chrome.tabs.get(tabId);
        const title = tab.title || 'No title found';
        console.log('URL:', url);
        console.log('Title:', title);

        // Lưu vào IndexedDB
        const pageData = {
            url: url,
            title: title,
            lastVisitTime: visitTime,
            is_embedded: false
        };
        const id = await storeHistoryItem(pageData);
        pageData.id = id;

        if (!isSearchURL(url)) {
            uploadHistory([pageData]);
        }
    }
});

function isValidURL(url) {
    try {
        new URL(url);
        return true;
    } catch (e) {
        return false;
    }
}

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

function isExcludedURL(url) {
    const excludedURLs = [
        'chrome://',
        'edge://',
        'about:',
        'chrome-extension://',
        'moz-extension://',
        'file://',
        'view-source:',
        'data:',
        'javascript:'
    ];

    return excludedURLs.some(excluded => url.startsWith(excluded)) || url.startsWith('chrome://');
}

function isRedirectURL(url) {
    const redirectDomains = [
        't.co',
        'bit.ly',
        'goo.gl',
        'tinyurl.com',
        'ow.ly',
        'is.gd',
        'buff.ly',
        'adf.ly',
        'tiny.cc',
        'lnkd.in',
        'db.tt',
        'qr.ae',
        'branch.io'
    ];

    try {
        const urlObj = new URL(url);
        return redirectDomains.some(domain => urlObj.hostname === domain || urlObj.hostname.endsWith('.' + domain));
    } catch (e) {
        console.error('Invalid URL:', url);
        return false;
    }
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
        const request = indexedDB.open('historyDB', 3);

        request.onupgradeneeded = event => {
            const db = event.target.result;
            const store = db.objectStoreNames.contains('history')
                ? event.target.transaction.objectStore('history')
                : db.createObjectStore('history', { keyPath: 'id', autoIncrement: true });

            if (!store.indexNames.contains('is_embedded')) {
                store.createIndex('is_embedded', 'is_embedded', { unique: false });
            }
            if (!store.indexNames.contains('line_id')) {
                store.createIndex('line_id', 'line_id', { unique: false });
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

async function updateProcessedItems(historyLines) {
    try {
        const db = await openDatabase();
        const tx = db.transaction('history', 'readwrite');
        const store = tx.objectStore('history');

        console.log('Updating processed items 1:', historyLines);

        // Flatten the nested array structure while preserving line_id
        const updatePromises = historyLines.flatMap(line => {
            console.log('Updating processed items 2:', line);
            
            // Each line is an array of items sharing the same line_id
            return line.map(async item => {
                try {
                    // Check if item exists
                    const existingItem = await get(store, item.id);
                    
                    if (existingItem) {
                        // Update existing item while preserving line_id
                        const updatedItem = {
                            ...existingItem,
                            ...item,
                            is_embedded: true,
                            line_id: item.line_id, // Ensure line_id is preserved
                            prev_item: item.prev_item,
                            next_item: item.next_item,
                        };
                        return put(store, updatedItem);
                    } else {
                        // Add new item with line_id
                        return add(store, {
                            ...item,
                            is_embedded: true,
                            line_id: item.line_id,
                            prev_item: item.prev_item,
                            next_item: item.next_item,
                        });
                    }
                } catch (error) {
                    console.error(`Error processing item ${item.id}:`, error);
                    throw error;
                }
            });
        });

        // Wait for all updates to complete
        await Promise.all(updatePromises);
        
        return new Promise((resolve, reject) => {
            tx.oncomplete = () => {
                console.log('Successfully updated all items with line_id information');
                resolve();
            };
            tx.onerror = () => reject(tx.error);
        });
    } catch (error) {
        console.error('Error in updateProcessedItems:', error);
        throw error;
    }
}

function get(store, id) {
    return new Promise((resolve, reject) => {
        const request = store.get(id);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

function put(store, item) {
    return new Promise((resolve, reject) => {
        const request = store.put(item);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

function add(store, item) {
    return new Promise((resolve, reject) => {
        const request = store.add(item);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
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
    } else if (message.action === 'deleteHistoryItem') {
        deleteHistoryItem(message.id).then(sendResponse);
        return true; // Keep the message channel open for the asynchronous response
    } else if (message.action === 'updateHistoryItem') {
        updateHistoryItem(message.item).then(sendResponse);
        return true; // Keep the message channel open for the asynchronous response
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getHistoryItem') {
        openDatabase().then(db => {
            const transaction = db.transaction(['history'], 'readonly');
            const objectStore = transaction.objectStore('history');

            let query;
            if (typeof request.idOrUrl === 'string' && request.idOrUrl.startsWith('http')) {
                query = objectStore.index('url').get(request.idOrUrl);
            } else {
                query = objectStore.get(request.idOrUrl);
            }

            query.onsuccess = (event) => {
                const item = event.target.result;
                if (item) {
                    sendResponse({ item: item });
                } else {
                    sendResponse({ error: 'Item not found' });
                }
            };

            query.onerror = (event) => {
                console.error('Error fetching history item:', event.target.error);
                sendResponse({ error: 'Error fetching history item' });
            };
        }).catch(error => {
            console.error('Error opening database:', error);
            sendResponse({ error: 'Error opening database' });
        });

        return true; // Indicates that the response is sent asynchronously
    }
    // ... (handle other actions)
});

async function storeHistoryItem(pageData) {
    const db = await openDatabase();
    const tx = db.transaction('history', 'readwrite');
    const store = tx.objectStore('history');

    // Check if the item already exists
    const existingItem = await get(store, pageData.url);
    
    if (existingItem) {
        // If the item exists, increment the visit count
        pageData.visitCount = (existingItem.visitCount || 0) + 1;
        return put(store, pageData);
    } else {
        // If it's a new item, set the initial visit count to 1
        pageData.visitCount = 1;
        return add(store, pageData);
    }
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