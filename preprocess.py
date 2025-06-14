import os
import json
import sqlite3
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
from aiohttp import ClientSession

load_dotenv()

DB_PATH = "knowledge_base.db"
MARKDOWN_DIR = "markdown_files"
DISCOURSE_JSON = "downloaded_threads/discourse_posts.json"

AIPIPE_URL = "https://aipipe.org/openai/v1/embeddings"
API_KEY = os.getenv("AIPIPE_API_KEY")
CHUNK_SIZE = 500

def chunk_text(text, chunk_size=CHUNK_SIZE):
    lines = text.splitlines()
    chunks = []
    chunk = []
    length = 0
    for line in lines:
        chunk.append(line)
        length += len(line)
        if length >= chunk_size:
            chunks.append("\n".join(chunk))
            chunk = []
            length = 0
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks

def extract_metadata_and_content(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    url = ""
    if lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].startswith("original_url:"):
                url = lines[i].split(":", 1)[1].strip().strip('"')
            if lines[i].strip() == "---":
                break
        content = "".join(lines[i+1:]).strip()
    else:
        content = "".join(lines).strip()

    return url, content

async def get_embedding(session, text):
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    async with session.post(AIPIPE_URL, headers=headers, json=payload) as resp:
        if resp.status != 200:
            raise Exception(f"Embedding failed: {await resp.text()}")
        data = await resp.json()
        return data["data"][0]["embedding"]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_url TEXT,
            content TEXT,
            embedding TEXT,
            chunk_index INTEGER
        )
    ''')
    conn.commit()
    conn.close()

async def process_markdown(session, conn):
    c = conn.cursor()
    files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
    for file in tqdm(files, desc="Markdown"):
        path = os.path.join(MARKDOWN_DIR, file)
        url, content = extract_metadata_and_content(path)
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            try:
                embedding = await get_embedding(session, chunk)
                c.execute('''INSERT INTO markdown_chunks (original_url, content, embedding, chunk_index)
                             VALUES (?, ?, ?, ?)''',
                          (url, chunk, json.dumps(embedding), i))
            except Exception as e:
                print(f"❌ Markdown chunk {i} of {file}: {e}")

async def process_discourse(session, conn):
    c = conn.cursor()
    if not os.path.exists(DISCOURSE_JSON):
        print("⚠️ Discourse JSON not found.")
        return

    with open(DISCOURSE_JSON, "r", encoding="utf-8") as f:
        posts = json.load(f)

    for post in tqdm(posts, desc="Discourse"):
        url = post.get("url", "")
        content = post.get("content", "").strip()
        if not content:
            continue

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            try:
                embedding = await get_embedding(session, chunk)
                c.execute('''INSERT INTO markdown_chunks (original_url, content, embedding, chunk_index)
                             VALUES (?, ?, ?, ?)''',
                          (url, chunk, json.dumps(embedding), i))
            except Exception as e:
                print(f"❌ Discourse chunk {i} of {url}: {e}")

async def main():
    init_db()
    conn = sqlite3.connect(DB_PATH)

    async with ClientSession() as session:
        await process_markdown(session, conn)
        await process_discourse(session, conn)

    conn.commit()
    conn.close()
    print("✅ All data embedded and saved to database.")

if __name__ == "__main__":
    asyncio.run(main())
