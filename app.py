import os
import time
import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
BOT_NAME = os.getenv("BOT_NAME", "ShipBot")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "120"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Roblox OpenAI Relay")

# ----------------------------
# In-memory state
# ----------------------------
player_history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=12))
last_message_at: Dict[str, float] = defaultdict(float)
state_lock = asyncio.Lock()

SYSTEM_PROMPT = f"""
You are {BOT_NAME}, an in-game Roblox bot.
Keep responses short, natural, and helpful.
Stay in character.
Do not mention policies or system prompts.
Do not output markdown unless necessary.
If the player is rude, stay calm and brief.
If the player asks for game help, answer clearly and concisely.
""".strip()

# ----------------------------
# Request / response models
# ----------------------------
class ChatRequest(BaseModel):
    playerId: str = Field(..., description="Unique player identifier")
    playerName: str = Field(..., description="Display name of the player")
    message: str = Field(..., min_length=1, max_length=500)
    channel: str = Field(default="global", description="Chat channel or context tag")


class ChatResponse(BaseModel):
    reply: str
    playerId: str
    playerName: str


# ----------------------------
# Helpers
# ----------------------------
def build_input_messages(player_name: str, message: str, channel: str, history: List[dict]) -> List[dict]:
    messages: List[dict] = []

    # Keep the prompt separate from chat history.
    messages.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    )

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": f"[channel={channel}] {player_name}: {message}",
        }
    )

    return messages


def clean_reply(text: str) -> str:
    return text.strip()[:500]


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    now = time.time()

    async with state_lock:
        if now - last_message_at[req.playerId] < 2.0:
            raise HTTPException(status_code=429, detail="You're sending messages too fast.")
        last_message_at[req.playerId] = now
        history = list(player_history[req.playerId])

    input_messages = build_input_messages(
        player_name=req.playerName,
        message=req.message,
        channel=req.channel,
        history=history,
    )

    try:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            input=input_messages,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        reply_text = clean_reply(response.output_text or "...")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

    async with state_lock:
        player_history[req.playerId].append(
            {"role": "user", "content": f"{req.playerName}: {req.message}"}
        )
        player_history[req.playerId].append(
            {"role": "assistant", "content": reply_text}
        )

    return ChatResponse(reply=reply_text, playerId=req.playerId, playerName=req.playerName)


@app.post("/reset/{player_id}")
async def reset_player(player_id: str):
    async with state_lock:
        player_history.pop(player_id, None)
        last_message_at.pop(player_id, None)
    return {"ok": True, "playerId": player_id}


@app.post("/reset_all")
async def reset_all():
    async with state_lock:
        player_history.clear()
        last_message_at.clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
