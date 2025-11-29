from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, json, base64, requests, hashlib, threading
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
import asyncio
import whisper
import torch
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import httpx
import time
from datetime import datetime
from configs.personality_loader import PersonalityLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
whisper_model = whisper.load_model("large-v3-turbo", device=device)
# GPUがあるときにパフォーマンスチューニング
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# 初期設定
# -----------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
AIVIS_URL = "http://127.0.0.1:10101"
HISTORY_FILE = "chat_history.json"
MAX_HISTORY = 10
BELLA_DATA = PersonalityLoader.load_personality("configs/personality_bella.json")
GROQ_MODEL = "llama-3.1-8b-instant"

BOT_PERSONALITY = BELLA_DATA.get("bot_personality", "")
speech_params = BELLA_DATA.get("speech_params", {})
BASE_PROMPT= BELLA_DATA.get("prompt", "")
#VOICE_CONFIG = BELLA_DATA.get("voice_settings", {})
SYSTEM_RULES = BELLA_DATA.get("system_rules", "")
STYLE_MAP = BELLA_DATA.get("STYLE_MAP", {})
EMOTION_PRESETS = BELLA_DATA.get("EMOTION_PRESETS", {})
REACTIONS = BELLA_DATA.get("REACTIONS", [])
if isinstance(REACTIONS, dict):
    REACTIONS = list(REACTIONS.values())

last_input_time = time.time()
last_screen_change_time = time.time()
last_voice_time = 0

trigger_10min_used = False
trigger_30min_used = False

TIME_TRIGGERS = {7, 12, 15, 19, 22, 23, 0, 2, 3, 4, 5}

current_screen_id = None

is_voice_playing = False

CACHE_DIR = Path("voice_cache")
CACHE_DIR.mkdir(exist_ok=True)

async def call_groq(messages, max_tokens=150, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.post(GROQ_API_URL, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
    
    except httpx.HTTPError as e:
        print("Groq HTTP error:", repr(e))
    except Exception as e:
        print("Groq error:", repr(e))

    return "ごめん、応答に失敗しちゃった..."

# -----------------------------
# 音声再生
# -----------------------------
def play_audio_async(wav_bytes):
    global is_voice_playing
    is_voice_playing = True
    
    def _play():
        data, samplerate = sf.read(BytesIO(wav_bytes))
        sd.play(data, samplerate)
        sd.wait()
        # 再生終了
        global is_voice_playing
        is_voice_playing = False

    threading.Thread(target=_play, daemon=True).start()

# -----------------------------
# 音声合成
# -----------------------------
async def synthesize_aivis_async(text: str, emotion: str = None):
    try:
        STYLE_ID = STYLE_MAP.get(emotion, speech_params.get("default_style_id", 888753763)) #Anneli
        cache_key = hashlib.md5(f"{STYLE_ID}_{text}_{emotion}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.wav"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
            
        # 感情ごとの音声パラメータを取得
        emotion_params = EMOTION_PRESETS.get(emotion, {}) if emotion else {}
        pitchScale = speech_params.get("pitch", -0.06) + emotion_params.get("pitchScale", 0.0)
        speedScale = speech_params.get("speed", 0.96) * emotion_params.get("speedScale", 1.0)
        intonationScale = speech_params.get("intonation", 0.9) * emotion_params.get("intonationScale", 1.0)

         # トーン微調整(emotionに応じて)
        if emotion:
            if "冷めてる" in emotion or "冷め" in emotion:
                pitchScale -= 0.1
                speedScale *= 0.9
            elif "強い" in emotion or "本気" in emotion:
                pitchScale += 0.2
                speedScale *= 1.1


        async with aiohttp.ClientSession() as session:
            async with session.post(f"{AIVIS_URL}/audio_query", params={"speaker": STYLE_ID, "text": text}) as qres:
                if qres.status != 200:
                    raise RuntimeError(f"audio_query faied: {qres.status}")
                query = await qres.json()

            # パラメータ調整
            query.update({
                "speedScale": round(speedScale, 2),
                "styleScale": speech_params.get("style_strength", 1.03),
                "intonationScale": round(intonationScale, 2),
                "pitchScale": round(pitchScale, 2),
                "volumeScale": speech_params.get("volume", 0.6),
                "prePhonemeLength": speech_params.get("pre_silence", 0.18),
                "postPhonemeLength": speech_params.get("post_silence", 0.25)
            })

            # synthesis
            async with session.post(f"{AIVIS_URL}/synthesis", params={"speaker": STYLE_ID}, json=query) as sres:
                if sres.status != 200:
                    raise RuntimeError(f"synthesis failed: {sres.status}")
                wav_bytes = await sres.read()

        #キャッシュ保存
        with open(cache_path, "wb") as f:
            f.write(wav_bytes)

        return base64.b64encode(wav_bytes).decode("utf-8")

    except Exception as e:
        print("AivisSpeech音声生成エラー:", repr(e))
        return None
    
def generate_system_prompt():
    # SYSTEM_RULES は dict の場合があるので文字列化
    rules_text = ""
    if isinstance(SYSTEM_RULES, dict):
        for k, v in SYSTEM_RULES.items():
            if isinstance(v, list):
                v = "、".join(v)
            rules_text += f"{k}: {v}\n"
    else:
        rules_text = str(SYSTEM_RULES)

    # personality は文字列 or 配列 or dict の可能性がある
    if isinstance(BOT_PERSONALITY, dict):
        persona_text = "\n".join([f"{k}: {v}" for k, v in BOT_PERSONALITY.items()])
    elif isinstance(BOT_PERSONALITY, list):
        persona_text = "\n".join(BOT_PERSONALITY)
    else:
        persona_text = str(BOT_PERSONALITY)

    return f"{persona_text}\n{rules_text}\n{BASE_PROMPT}"


# =========================================
# ローカル感情分析（純関数）
# =========================================
async def analyze_emotion_local(text: str) -> str:
    prompt = f"""
    以下の日本語の発話に最も強く現れる「感情」を１つだけ短く返してください。
    返答は以下のいずれかとします：

    喜び / 楽しさ / 安心 / 興奮 / 悲しみ / 苦痛 / 不安 / 怒り / 驚き / 無関心 / 疲れ / 困惑 / 恐れ

    返答は必ず「感情名」だけにしてください。理由や説明は禁止。
    入力：{text}
    出力：
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "あなたは日本語の感情分析アシスタントです。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 50,
    }

    async with httpx.AsyncClient(timeout=6.0) as client:

        res = await client.post(GROQ_API_URL, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()

# -----------------------------
# FastAPI 初期化
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# 音声再生ウォッチドッグ
# -----------------------------
def start_voice_watchdog():
    def loop():
        while True:
            msg = check_voice_trigger()
            if msg:
                asyncio.run(play_voice_message(msg))
            time.sleep(5)
    threading.Thread(target=loop, daemon=True).start()

async def play_voice_message(text):
    wav_b64 = await synthesize_aivis_async(text)
    if wav_b64:
        wav_bytes = base64.b64decode(wav_b64)
        play_audio_async(wav_bytes)

# -----------------------------
# キャッシュ辞書
# -----------------------------
reaction_cache = {}

@app.on_event("startup")
async def startup_event():
    # リアクション音声キャッシュ作成
    for file in CACHE_DIR.glob("*.wav"):
        key = file.stem
        with open(file, "rb") as f:
            reaction_cache[key] = base64.b64encode(f.read()).decode("utf-8")

    start_voice_watchdog() # 音声再生ウォッチドッグ開始

# -----------------------------
# 音声トリガーチェック
# -----------------------------
def on_user_input():
    global last_input_time
    last_input_time = time.time()

def on_screen_change(screen_id):
    global current_screen_id, last_screen_change_time
    if screen_id != current_screen_id:
        current_screen_id = screen_id
        last_screen_change_time = time.time()

def check_voice_trigger():
    global last_voice_time, trigger_10min_used, trigger_30min_used

    now = time.time()
    dt = datetime.now()

    # === 時間トリガー ===
    if dt.minute == 0 and dt.hour in TIME_TRIGGERS:
        if now - last_voice_time > 300:  # 5分以内に発話してなければOK
            last_voice_time = now
            return f"{dt.hour}時だよー。どう？ 無理してない？"

    # === 操作なし時間の計算 ===
    idle_time = min(now - last_input_time, now - last_screen_change_time)

    # === 最初の10分 idle ===
    if idle_time > 600 and not trigger_10min_used:
        trigger_10min_used = True
        last_voice_time = now
        return "10分くらい操作ないけど大丈夫？ ちょっと休憩？"

    # === 最初の30分 idle ===
    if idle_time > 1800 and not trigger_30min_used:
        trigger_30min_used = True
        last_voice_time = now
        return "30分くらい動きがないね。大丈夫？ 何か詰まってる？"

    # === 継続 idle：最後の発話から1時間 ===
    if idle_time > 600 and now - last_voice_time > 3600:
        last_voice_time = now
        return "ずっと操作ないみたいだけど大丈夫？ ちょっと休もう？"

    return None

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
    if not audio_base64:
        audio_base64 = await synthesize_aivis_async(chosen)
        if audio_base64:
            reaction_cache[chosen] = audio_base64
    return {"text": chosen, "audio": audio_base64}


# -----------------------------
# Whisper モデルロード
# -----------------------------
def clean_audio(input_path: str, output_path: str):
    audio = AudioSegment.from_file(input_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    reduced = nr.reduce_noise(y=samples, sr=audio.frame_rate)
    cleaned = AudioSegment(
        reduced.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    cleaned.export(output_path, format="wav")
    

def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(
        file_path,
        language="ja",         # 日本語を明示的に指定（自動検出より正確）
        task="transcribe",     # 翻訳ではなく文字起こしを指定
        temperature=0,         # 出力のブレを減らす（誤字を減らす）
        fp16=True,             # GPU利用時に高速化（GPUが使えているなら有効）
        condition_on_previous_text=False,  # VAD動作強化
        word_timestamps=False, # 単語ごとのタイムスタンプ不要
        verbose=False          # ログを抑制（必要ならTrueに）
    )
    return result["text"].strip()


# -----------------------------
# /chat エンドポイント
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_message = data.get("text")

    if not user_message:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    chat_history.append({"role": "user", "content": user_message})
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    system_prompt = generate_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.7, "max_tokens": 512}

    bot_reply = "...ごめん。今ちょっと応答ができなかったみたい。"

    try:
        bot_reply = await call_groq(messages)
    except Exception as e:
        print("chat LLM error:", repr(e))
    
    # 🔹 感情分析を実行
    emotion = "落ち着き"
    try:
        emotion = (await analyze_emotion_local(bot_reply)).split(" (")[0]
    except Exception as e:
        print("emotion analysis error:", repr(e))

    # 🔹 感情トーンを反映して音声生成
    audio_base64 = await synthesize_aivis_async(bot_reply, emotion)

    chat_history.append({"role": "assistant", "content": bot_reply})
    save_history()

    return JSONResponse({"reply": bot_reply,"emotion": emotion, "audio": audio_base64})

# -----------------------------
# /whisper エンドポイント
# -----------------------------
@app.post("/whisper")
async def whisper_endpoint(file: UploadFile = File(...)):
    if is_voice_playing:
        return JSONResponse({"user_text": "", "replay": "", "emotion_ai": "無効", "aydio": None})
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    #whisperで文字起こし
    try:
        user_text = transcribe_audio(temp_path)
    finally:
        os.remove(temp_path)
    
    # 会話履歴に追加
    chat_history.append({"role": "user", "content": user_text})
    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    
    #LLM応答取得
    system_prompt = generate_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.7, "max_tokens": 512}

    bot_reply = "⚠応答の取得に失敗しました"
    try:
        bot_reply = await call_groq(messages)
    except Exception as e:
        print("whisper LLM error:", repr(e))

    #感情分析
    emotion_ai = "落ち着き"
    try:
        emotion_ai = (await analyze_emotion_local(bot_reply)).split(" (")[0] 
    except Exception as e:
        print("emotion analysis error:", repr(e))

    # AI返答の感情
    audio_base64 = await synthesize_aivis_async(bot_reply, emotion_ai)

    return JSONResponse({
        "user_text": user_text,
        "reply": bot_reply,
        "emotion_ai": emotion_ai,
        "audio": audio_base64
    })

#voice_stream エンドポイント
@app.post("/voice_stream")
async def voice_stream(request: Request):
    data = await request.json()
    text = data["text"]

    #synthesize_aivis_asyncを使って音声生成
    audio_base64 = await synthesize_aivis_async(text, emotion="落ち着き")
    wav_bytes = base64.b64decode(audio_base64)

    async def audio_generator():
        yield wav_bytes

    return StreamingResponse(audio_generator(), media_type="audio/wav")

#感情分析エンドポイント
@app.post("/analyze_emotion")
async def analyze_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")

    prompt = f"""
    以下の日本語の発話を読み取り、話者の「感情」と「トーン（本気・冷めてる・皮肉など）」を短く表してください。
    出力形式：「感情（トーン）」
    例：
    「うれしいです！」 → 喜び（本気）
    「うわー、すごくうれしいです」 → 喜び（強い）
    「うわー、すごくうれしいです（棒読み）」 → 喜び（冷めてる）
    「はぁ…うれしいです」 → 喜び（疲れ気味）
    「べつにうれしくないけど」 → 無関心（皮肉）

    入力：{text}
    出力：
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "あなたは日本語の感情分析アシスタントです。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 50,
    }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
          res = await client.post(GROQ_API_URL, json=payload, headers=headers)
          res.raise_for_status()
          data = res.json()
          result = data["choices"][0]["message"]["content"].strip()
          return {"emotion": result}
    except Exception as e:
        print("emotion analysis error:", e)
        return {"emotion": "中立（エラー）"}