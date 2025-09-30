from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# 会話履歴（簡易版）
chat_history = []

# 性格設定
BOT_PERSONALITY = os.getenv("BOT_PERSONALITY", "あなたはフレンドリーな会話パートナーです。")

@app.get("/", response_class=FileResponse)
async def get_chat_page():
    return FileResponse("chat.html")

MAX_HISTORY = 10  # ユーザー発言10件＋Bot返信10件を保持したい場合は調整

@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_message = data.get("message")
    
    # ユーザー発言を追加
    chat_history.append({"role": "user", "content": user_message})
    
    # 履歴を直近 MAX_HISTORY 件のペアに制御
    # ユーザーとBotのペアを1セットとして扱う
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)  # 古いものから削除

    messages = [{"role": "system", "content": BOT_PERSONALITY}] + chat_history

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    bot_reply = response.json()["choices"][0]["message"]["content"]

    # Bot発言を追加
    chat_history.append({"role": "assistant", "content": bot_reply})

    return {"reply": bot_reply}

