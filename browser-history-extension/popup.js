document.addEventListener('DOMContentLoaded', () => {

    const chatInput = document.getElementById('chatInput');
    const chatButton = document.getElementById('chatButton');
    const chatResponse = document.getElementById('chatResponse');
    const historyButton = document.getElementById('historyButton');
    const assistantHeader = document.getElementById('assistantHeader');

    chatButton.addEventListener('click', async () => {
        const message = chatInput.value.trim();
        if (message) {
            const response = await fetch('http://localhost:5000/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message })
            });

            console.log('Response:', response);

            const messageElement = document.createElement('div');
            messageElement.textContent = response;
            chatResponse.appendChild(messageElement);

            chatInput.value = '';

            const headerH2 = assistantHeader.querySelector('h2');
            if (headerH2) {
                headerH2.remove();
            }
        }
    });

    historyButton.addEventListener('click', () => {
        window.open('history.html', '_blank');
    });

    loadChatHistory();
});
