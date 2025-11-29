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
  const response = await fetch("/voice_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  // MediaSourceを使って受信しながら再生
  const mediaSource = new MediaSource();
  const audio = new Audio();
  audio.src = URL.createObjectURL(mediaSource);
  audio.play();

  mediaSource.addEventListener("sourceopen", async () => {
    const sourceBuffer = mediaSource.addSourceBuffer('audio/wav; codecs="1"');
    const reader = response.body.getReader();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sourceBuffer.appendBuffer(value);
    }

    mediaSource.endOfStream();
  });
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
    const audioBlob = base64ToBlob(data.audio, "audio/wav");
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
    }

    // Base64 → Blob
    function base64ToBlob(base64, type = "audio/wav") {
        const binary = atob(base64);
        const len = binary.length;
        const buffer = new Uint8Array(len);
        for (let i = 0; i < len; i++) buffer[i] = binary.charCodeAt(i);
        return new Blob([buffer], { type });
    }


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
