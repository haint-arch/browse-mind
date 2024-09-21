if (typeof $ === 'undefined') {
    console.error('jQuery is not loaded');
} else {
    console.log('jQuery is loaded successfully');
}

document.addEventListener('DOMContentLoaded', async () => {
    const db = await openDatabase();
    const tx = db.transaction('history', 'readonly');
    const store = tx.objectStore('history');

    const historyTable = document.getElementById('historyTable').querySelector('tbody');

    const downloadButton = document.getElementById('downloadJsonBtn');

    // Sự kiện click vào nút tải xuống
    downloadButton.addEventListener('click', async () => {
        const jsonData = await getAllHistoryData(store);
        const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'browsing_history.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    store.openCursor().onsuccess = event => {
        const cursor = event.target.result;
        if (cursor) {
            const { url, visitTime } = cursor.value;
            addHistoryItemToTable(historyTable, url, visitTime);
            cursor.continue();
        }
    };

    browser.runtime.onMessage.addListener((message) => {
        console.log('Message received in history.js:', message);
        if (message.action === 'newHistoryItem') {
            addHistoryItemToTable(historyTable, message.item.url, message.item.visitTime);
        }
    });

    historyTable.addEventListener('click', async (event) => {
        if (event.target.classList.contains('history-link')) {
            console.log('History link clicked:', event.target);
            event.preventDefault();
            const url = event.target.getAttribute('data-url');
            console.log('Opening page details for:', url);
            
            const details = await getPageDetails(url);

            document.getElementById('pageTitle').textContent = details.title;
            document.getElementById('pageContent').textContent = details.content;
            document.getElementById('pageColors').textContent = details.colors.join(', ');

            console.log("Showing modal...");
            $('#detailsModal').modal('show');
        }
    });
});



function addHistoryItemToTable(historyTable, url, visitTime) {
    const displayUrl = url.length > 50 ? url.substring(0, 50) + '...' : url;

    const row = document.createElement('tr');
    row.innerHTML = `
        <td>
            <a href="#" class="history-link small text-truncate d-inline-block" style="max-width: 200px;" data-url="${url}" title="${url}">
                ${displayUrl}
            </a>
        </td>
        <td class="small">${new Date(visitTime).toLocaleString()}</td>
    `;
    historyTable.prepend(row); // Thêm phần tử mới lên đầu danh sách
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

function getPageDetails(url) {
    return new Promise((resolve) => {
        browser.runtime.sendMessage({ action: 'getPageDetails', url }, resolve);
    });
}

async function getAllHistoryData(store) {
    return new Promise((resolve, reject) => {
        const allData = [];
        store.openCursor().onsuccess = event => {
            const cursor = event.target.result;
            if (cursor) {
                allData.push(cursor.value);
                cursor.continue();
            } else {
                resolve(allData);
            }
        };
        store.openCursor().onerror = event => reject(event.target.error);
    });
}

browser.runtime.onMessage.addListener((message) => {
    console.log('Message received in history.js:', message);
    if (message.action === 'newHistoryItem') {
        addHistoryItemToTable(historyTable, message.item.url, message.item.visitTime);
    }
});
