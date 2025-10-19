import { BellaCore } from '/static/core.js';

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');

let bella = null;
let recognizing = false;
let recognition = null;

// BellaCore 初期化
(async () => {
    bella = await BellaCore.getInstance();
    bella.playIdle();
})();

// メッセージ追加
function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}
// 音声再生
async function playVoice(text) {
  const res = await fetch("/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}


// 音声読み上げ
function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ja-JP';
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
}

// メッセージ送信処理
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || !bella) return;

    addMessage('user', text);
    inputEl.value = '';

    bella.playListening();

    const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
    });

    const data = await res.json();

    bella.playTalking();

    addMessage('assistant', data.reply);

    // 音声が含まれている場合は再生
    if (data.audio) {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
        audio.play();
    }
    //speakText(data.reply);
    //playVoice(data.reply); // こちらを使う場合はコメントアウトを外す

    setTimeout(() => bella.playIdle(), 4000);
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// 🎤 音声認識の初期化
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.lang = 'ja-JP';
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.onstart = () => {
        recognizing = true;
        micBtn.textContent = '🛑';
        bella.playListening();
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        inputEl.value = transcript;
        sendMessage();
    };

    recognition.onerror = (e) => {
        console.error('音声認識エラー:', e);
    };

    recognition.onend = () => {
        recognizing = false;
        micBtn.textContent = '🎤';
        bella.playIdle();
    };

    micBtn.addEventListener('click', () => {
        if (recognizing) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });
} else {
    micBtn.disabled = true;
    micBtn.textContent = '🎤(非対応)';
}
