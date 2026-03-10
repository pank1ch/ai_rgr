import json
import asyncio
from fastapi import FastAPI
from telethon import TelegramClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TELEGRAM_CHANNELS = [
    "@tboom_54",
    "https://t.me/+MZsPh4nOVPRjMzhi"
]

USER_PROMPT = """
Извлеки информацию о перелете из текста.

Верни JSON строго в формате:

{
 "departure_city": "",
 "arrival_city": "",
 "round_trip": true/false,
 "price": "",
 "departure_date": ""
}

Если данных нет — ставь null.
"""

api_id = 123456789
api_hash = "api_hash"

client = TelegramClient("session", api_id, api_hash)

MODEL_NAME = "ai-forever/ruT5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

app = FastAPI()

async def collect_messages(limit=10):

    await client.start()

    messages = []

    for channel in TELEGRAM_CHANNELS:

        print("Collecting from:", channel)

        async for msg in client.iter_messages(channel, limit=limit):

            if msg.text:
                messages.append({
                    "channel": channel,
                    "text": msg.text
                })

        await asyncio.sleep(1)

    return messages

def run_llm(post_text):

    prompt = f"""
{USER_PROMPT}

Текст:
{post_text}

JSON:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        return json.loads(result_text)
    except:
        return {"raw_output": result_text}


@app.get("/run")
async def run_pipeline():

    posts = await collect_messages(limit=10)

    results = []

    for post in posts:

        extracted = run_llm(post["text"])

        results.append({
            "channel": post["channel"],
            "original_text": post["text"],
            "extracted": extracted
        })

    return results