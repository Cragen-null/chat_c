from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
import tempfile
import json

HISTORY_FILE = "chat_history.json"

# 履歴をファイルに保存
def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

# 履歴をファイルから読み込み
def load_history():
    global chat_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

# 起動時に履歴をロード
load_history()
   
load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==============================
# 基本設定
# ==============================
AIVIS_URL = "http://127.0.0.1:10101"

AIVIS_SPEAKER = "0"  # ボイスID（0など、エンジンの出力で確認）

# ==============================
# 音声合成関数（AivisSpeech用）
# ==============================
def synthesize_aivis(text: str):
    try:
        STYLE_ID = "888753763"
        params = personality_data.get("speech_params", {})

        # クエリ生成
        query_response = requests.post(
            f"{AIVIS_URL}/audio_query",
            params={"text": text, "speaker": STYLE_ID}
        )
        query_response.raise_for_status()
        query = query_response.json()

        # personalityからパラメータを適用
        query.update({
            "speedScale": speech_params.get("speed", 0.96),
            "styleScale": speech_params.get("style_strength", 1.03),
            "intonationScale": speech_params.get("intonation", 0.9),
            "pitchScale": speech_params.get("pitch", -0.06),
            "volumeScale": speech_params.get("volume", 0.6),
            "prePhonemeLength": speech_params.get("pre_silence", 0.18),
            "postPhonemeLength": speech_params.get("post_silence", 0.25)
        })

        synth_response = requests.post(
            f"{AIVIS_URL}/synthesis",
            params={"speaker": STYLE_ID},
            data=json.dumps(query, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        synth_response.raise_for_status()
        return base64.b64encode(synth_response.content).decode("utf-8")

    except Exception as e:
        print("AivisSpeech音声生成エラー:", e)
        return None


# 会話履歴（簡易版）
chat_history = []

# 性格設定
BOT_PERSONALITY_PATH = os.getenv("BOT_PERSONALITY_PATH", "configs/personality_bella.json")
with open(BOT_PERSONALITY_PATH, "r", encoding="utf-8") as f:
    personality_data = json.load(f)

# systemメッセージ用文字列に変換
traits_str = "\n- ".join(personality_data.get("personality_traits", []))
policy_str = "\n- ".join(personality_data.get("conversation_policy", []))
knowledge_str = "\n- ".join(personality_data.get("knowledge_scope", []))

BOT_PERSONALITY = f"""
あなたの名前は {personality_data['name']} です。
{personality_data['greeting']}

【性格】
- {traits_str}

【会話方針】
- {policy_str}

【知識範囲】
- {knowledge_str}

最初の発話では、必ず自己紹介を行ってください。
"""

# 音声合成関数ではspeech_paramsを参照
speech_params = personality_data.get("speech_params", {})


# 現在のスクリプトのディレクトリ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 静的ファイルをマウント
#app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ルート（index.html）
@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))
    #return FileResponse("chat.html") #単独動作テスト用
    
MAX_HISTORY = 10  # ユーザー発言10件＋Bot返信10件を保持したい場合は調整

@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_message = data.get("message")

    # ユーザー発言を履歴に追加
    chat_history.append({"role": "user", "content": user_message})
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    messages = [{
    "role": "system",
    "content": f"あなたは {personality_data['name']} というキャラクターです。"
               f"口調: {personality_data['tone']}、"
               f"性格: {personality_data['style']}。\n"
               f"以下は初期メッセージ例です: {personality_data['greeting']}\n\n"
               f"{BOT_PERSONALITY}"
}] + chat_history
    

    # ✅ ここが重要！（関数の中に入れる）
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

    # 🔹 Groq API呼び出し
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    bot_reply = response.json()["choices"][0]["message"]["content"]

    # 🔹 Bot発言を履歴に追加
    chat_history.append({"role": "assistant", "content": bot_reply})
    save_history()  # 履歴をファイルに保存

    # 🔹 音声生成
    audio_base64 = synthesize_aivis(bot_reply)

    return JSONResponse({
        "reply": bot_reply,
        "audio": audio_base64
    })



@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")
