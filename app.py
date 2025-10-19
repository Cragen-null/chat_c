from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import requests, os, json, base64, whisper, torch
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# -----------------------------
# 初期設定
# -----------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
AIVIS_URL = "http://127.0.0.1:10101"
BOT_PERSONALITY_PATH = os.getenv("BOT_PERSONALITY_PATH", "configs/personality_bella.json")
HISTORY_FILE = "chat_history.json"
MAX_HISTORY = 10

# -----------------------------
# 性格データ読み込み
# -----------------------------
with open(BOT_PERSONALITY_PATH, "r", encoding="utf-8") as f:
    personality_data = json.load(f)

speech_params = personality_data.get("speech_params", {})

# systemメッセージ用文字列
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

# -----------------------------
# 音声合成関数
# -----------------------------
def synthesize_aivis(text: str):
    try:
        STYLE_ID = "888753763"

        # audio_query作成
        query_response = requests.post(
            f"{AIVIS_URL}/audio_query",
            params={"text": text, "speaker": STYLE_ID}
        )
        query_response.raise_for_status()
        query = query_response.json()

        # personalityのパラメータ適用
        query.update({
            "speedScale": speech_params.get("speed", 0.96),
            "styleScale": speech_params.get("style_strength", 1.03),
            "intonationScale": speech_params.get("intonation", 0.9),
            "pitchScale": speech_params.get("pitch", -0.06),
            "volumeScale": speech_params.get("volume", 0.6),
            "prePhonemeLength": speech_params.get("pre_silence", 0.18),
            "postPhonemeLength": speech_params.get("post_silence", 0.25)
        })

        # synthesis
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

# -----------------------------
# リアクション音プリ生成
# -----------------------------
REACTIONS = ["はい", "う～ん", "えっと", "ん〜", "そうですね", "なるほど", "了解", "わかりました","えへへ"]
reaction_cache = {}

for text in REACTIONS:
    audio_base64 = synthesize_aivis(text)
    if audio_base64:
        reaction_cache[text] = audio_base64

# -----------------------------
# FastAPI初期化
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# 会話履歴
# -----------------------------
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
else:
    chat_history = []

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

# -----------------------------
# ルート
# -----------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# -----------------------------
# リアクション音取得
# -----------------------------
@app.get("/reaction")
async def reaction(text: str = None):
    import random
    chosen = text or random.choice(REACTIONS)
    audio_base64 = reaction_cache.get(chosen)
    return JSONResponse({"text": chosen, "audio": audio_base64})

# -----------------------------
# Whisperモデルロード
# -----------------------------
torch.set_num_threads(os.cpu_count())
whisper_model = whisper.load_model("base")

def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(file_path)
    return result["text"]

# -----------------------------
# /chat エンドポイント
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_message = data.get("message")

    chat_history.append({"role": "user", "content": user_message})
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    messages = [{
        "role": "system",
        "content": BOT_PERSONALITY
    }] + chat_history

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.7, "max_tokens": 512}

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    bot_reply = response.json()["choices"][0]["message"]["content"]
    
    chat_history.append({"role": "assistant", "content": bot_reply})
    save_history()

    audio_base64 = synthesize_aivis(bot_reply)

    return JSONResponse({"reply": bot_reply, "audio": audio_base64})

# -----------------------------
# /whisper エンドポイント
# -----------------------------
@app.post("/whisper")
async def whisper_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        user_text = transcribe_audio(temp_path)
    finally:
        os.remove(temp_path)

    chat_history.append({"role": "user", "content": user_text})
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    messages = [{"role": "system", "content": BOT_PERSONALITY}] + chat_history

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.7, "max_tokens": 512}

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    bot_reply = response.json()["choices"][0]["message"]["content"]

    chat_history.append({"role": "assistant", "content": bot_reply})
    save_history()

    audio_base64 = synthesize_aivis(bot_reply)

    return JSONResponse({"user_text": user_text, "reply": bot_reply, "audio": audio_base64})
