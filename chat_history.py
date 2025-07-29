# import libraries
import sqlite3
from datetime import datetime
from pathlib import Path

# Ensure database file exists
DB_FILE = "chat_history.db"
Path(DB_FILE).touch(exist_ok=True)

# Initialize DB and table if not exists
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

def save_chat_to_db(file_name, query, answer):
    timestamp = datetime.now().isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (file_name, query, answer, timestamp) VALUES (?, ?, ?, ?)",
            (file_name, query, answer, timestamp)
        )
        conn.commit()

def get_chat_history(file_name):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT query, answer, timestamp FROM chat_history WHERE file_name = ? ORDER BY id ASC", (file_name,))
        rows = cursor.fetchall()
        return [
            {"query": row[0], "answer": row[1], "timestamp": row[2]}
            for row in rows
        ]