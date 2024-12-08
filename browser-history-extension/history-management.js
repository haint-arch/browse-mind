document.addEventListener('DOMContentLoaded', () => {
    const categorySelect = document.getElementById('category');
    const lineIdSelect = document.getElementById('line-id');
    const searchInput = document.getElementById('search');
    const historyTableBody = document.getElementById('history-table-body');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageInfo = document.getElementById('page-info');
    const popup = document.getElementById('popup');
    const popupContent = document.getElementById('popup-content');
    const popupClose = document.querySelector('.popup-close');
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const loadingOverlay = document.querySelector('.loading-overlay');

    let currentPage = 1;
    const itemsPerPage = 10;
    let totalItems = 0;
    let filteredHistory = [];
    let allHistoryItems = [];

    function fetchHistoryFromIndexedDB() {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({ action: 'getAllHistoryItems' }, (response) => {
                if (response.error) {
                    reject(response.error);
                } else {
                    resolve(response.items);
                }
            });
        });
    }

    function populateCategories(historyItems) {
        const categories = new Set();
        const lineIds = new Set();
        historyItems.forEach(item => {
            if (item.categories) {
                item.categories.forEach(category => categories.add(category));
            }
            if (item.line_id) {
                lineIds.add(item.line_id);
            }
        });
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            categorySelect.appendChild(option);
        });
        lineIds.forEach(lineId => {
            const option = document.createElement('option');
            option.value = lineId;
            option.textContent = lineId;
            lineIdSelect.appendChild(option);
        });
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    async function applyFilters() {
        showLoading();
        const categoryFilter = categorySelect.value;
        const lineIdFilter = lineIdSelect.value;
        const searchFilter = searchInput.value.trim().toLowerCase();
        const startDate = startDateInput.value ? new Date(startDateInput.value).getTime() : 0;
        const endDate = endDateInput.value ? new Date(endDateInput.value).getTime() + 86400000 : Infinity;

        filteredHistory = allHistoryItems.filter(item => {
            const categoryMatch = !categoryFilter || (item.categories && item.categories.includes(categoryFilter));
            const lineIdMatch = !lineIdFilter || item.line_id === lineIdFilter;
            const searchMatch = !searchFilter || 
                item.title.toLowerCase().includes(searchFilter) || 
                item.url.toLowerCase().includes(searchFilter);
            const timeMatch = item.lastVisitTime >= startDate && item.lastVisitTime <= endDate;
            return categoryMatch && lineIdMatch && searchMatch && timeMatch;
        });

        totalItems = filteredHistory.length;
        currentPage = 1;
        await new Promise(resolve => setTimeout(resolve, 300)); // Simulate delay
        updateTable();
        updatePagination();
        hideLoading();
    }

    const debouncedApplyFilters = debounce(applyFilters, 300);

    function truncateUrl(url, maxLength = 30) {
        if (url.length <= maxLength) return url;
        return url.substring(0, maxLength - 3) + '...';
    }

    function updateTable() {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const pageItems = filteredHistory.slice(startIndex, endIndex);

        historyTableBody.innerHTML = '';
        pageItems.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.title}</td>
                <td><a href="${item.url}" target="_blank" class="truncate" title="${item.url}">${truncateUrl(item.url)}</a></td>
                <td>${item.categories ? item.categories.join(', ') : ''}</td>
                <td>${item.line_id || ''}</td>
                <td>${new Date(item.lastVisitTime).toLocaleString()}</td>
                <td class="action-column">
                    <div class="action-buttons">
                        <button class="edit-btn" title="Sửa">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
                                <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
                            </svg>
                        </button>
                        <button class="delete-btn" title="Xóa">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
                                <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                            </svg>
                        </button>
                    </div>
                </td>
            `;
            row.querySelector('.edit-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                editRecord(item);
            });
            row.querySelector('.delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteRecord(item);
            });
            row.addEventListener('click', (e) => {
                if (!e.target.closest('.edit-btn') && !e.target.closest('.delete-btn')) {
                    showPopup(item);
                }
            });
            historyTableBody.appendChild(row);
        });
    }

    function updatePagination() {
        const totalPages = Math.ceil(totalItems / itemsPerPage);
        pageInfo.textContent = `Trang ${currentPage} / ${totalPages}`;
        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages;
    }

    function showPopup(item) {
        popupContent.innerHTML = `
            <p><strong>Tiêu đề:</strong> ${item.title}</p>
            <p><strong>URL:</strong> <a href="${item.url}" target="_blank">${item.url}</a></p>
            <p><strong>Danh mục:</strong> ${item.categories ? item.categories.join(', ') : 'Không có'}</p>
            <p><strong>Line ID:</strong> ${item.line_id || 'Không có'}</p>
            <p><strong>Thời gian truy cập:</strong> ${new Date(item.lastVisitTime).toLocaleString()}</p>
            <p><strong>Số lần truy cập:</strong> ${item.visitCount || 'Không có thông tin'}</p>
            ${item.color ? `<p><strong>Màu sắc:</strong> <span style="background-color: rgb(${item.color.join(',')}); padding: 2px 10px; border-radius: 3px;">rgb(${item.color.join(',')})</span></p>` : ''}
            ${item.score !== undefined ? `<p><strong>Điểm:</strong> ${item.score.toFixed(4)}</p>` : ''}
        `;
        popup.style.display = 'block';
    }

    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            updateTable();
            updatePagination();
        }
    });

    nextPageButton.addEventListener('click', () => {
        const totalPages = Math.ceil(totalItems / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            updateTable();
            updatePagination();
        }
    });

    popupClose.addEventListener('click', () => {
        popup.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === popup) {
            popup.style.display = 'none';
        }
    });

    function setDefaultDates() {
        const today = new Date();
        const oneMonthAgo = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
        
        endDateInput.valueAsDate = today;
        startDateInput.valueAsDate = oneMonthAgo;
    }

    // Add event listeners for real-time filtering
    categorySelect.addEventListener('change', applyFilters);
    lineIdSelect.addEventListener('change', applyFilters);
    startDateInput.addEventListener('change', applyFilters);
    endDateInput.addEventListener('change', applyFilters);
    searchInput.addEventListener('input', debouncedApplyFilters);

    showLoading();
    fetchHistoryFromIndexedDB()
        .then(historyItems => {
            allHistoryItems = historyItems;
            filteredHistory = historyItems;
            totalItems = historyItems.length;
            populateCategories(historyItems);
            updateTable();
            updatePagination();
            hideLoading();
            setDefaultDates();
        })
        .catch(error => {
            console.error('Error fetching history:', error);
            hideLoading();
            historyTableBody.innerHTML = '<tr><td colspan="5">Error loading history. Please try again.</td></tr>';
        });

    function editRecord(item) {
        const editForm = document.createElement('form');
        editForm.innerHTML = `
            <label for="edit-title">Tiêu đề:</label>
            <input type="text" id="edit-title" value="${item.title}" required><br>
            <label for="edit-url">URL:</label>
            <input type="url" id="edit-url" value="${item.url}" required><br>
            <label for="edit-categories">Danh mục (phân cách bằng dấu phẩy):</label>
            <input type="text" id="edit-categories" value="${item.categories ? item.categories.join(', ') : ''}"><br>
            <label for="edit-line-id">Line ID:</label>
            <input type="text" id="edit-line-id" value="${item.line_id || ''}"><br>
            <label for="edit-last-visit-time">Thời gian truy cập:</label>
            <input type="datetime-local" id="edit-last-visit-time" value="${new Date(item.lastVisitTime).toISOString().slice(0, 16)}" required><br>
            <label for="edit-color">Màu sắc (R,G,B):</label>
            <input type="text" id="edit-color" value="${item.color ? item.color.join(',') : ''}"><br>
            <button type="submit">Lưu</button>
        `;
        
        editForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const updatedItem = {
                ...item,
                title: document.getElementById('edit-title').value,
                url: document.getElementById('edit-url').value,
                categories: document.getElementById('edit-categories').value.split(',').map(cat => cat.trim()).filter(cat => cat),
                line_id: document.getElementById('edit-line-id').value,
                lastVisitTime: new Date(document.getElementById('edit-last-visit-time').value).getTime(),
                color: document.getElementById('edit-color').value.split(',').map(Number),
            };
            updateRecord(updatedItem);
        });
        
        popupContent.innerHTML = '';
        popupContent.appendChild(editForm);
        popup.style.display = 'block';
    }

    function updateRecord(updatedItem) {
        chrome.runtime.sendMessage({ action: 'updateHistoryItem', item: updatedItem }, (response) => {
            if (response.success) {
                const index = allHistoryItems.findIndex(item => item.id === updatedItem.id);
                if (index !== -1) {
                    allHistoryItems[index] = updatedItem;
                    applyFilters();
                }
                popup.style.display = 'none';
            } else {
                alert('Không thể cập nhật bản ghi. Vui lòng thử lại.');
            }
        });
    }

    function deleteRecord(item) {
        if (confirm('Bạn có chắc chắn muốn xóa bản ghi này?')) {
            chrome.runtime.sendMessage({ action: 'deleteHistoryItem', id: item.id }, (response) => {
                if (response.success) {
                    allHistoryItems = allHistoryItems.filter(i => i.id !== item.id);
                    applyFilters();
                } else {
                    alert('Không thể xóa bản ghi. Vui lòng thử lại.');
                }
            });
        }
    }
});