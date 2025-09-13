import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import httpx
from model import get_fingerprint_embedding
from enhance import enhance_fingerprint

DB_PATH = "data.db"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DB UTILS ==================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row  # rows as dict-like
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS person (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        phone TEXT,
        role TEXT,
        address TEXT,
        age INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fingerprint (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        finger TEXT,
        FOREIGN KEY(person_id) REFERENCES person(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS scans_vec USING vec0(
        id INTEGER PRIMARY KEY,
        embedding FLOAT[512] DISTANCE COSINE
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ================== SCHEMAS ==================
class PersonIn(BaseModel):
    name: str
    email: str
    phone: str
    role: str
    address: str
    age: int

class PersonOut(PersonIn):
    id: int
    status: str

class ScanIn(BaseModel):
    person_id: int
    finger: str

class ScanOut(ScanIn):
    id: int

# ================== ROUTES ==================
@app.post("/api/people", response_model=PersonOut, status_code=201)
def create_person(person: PersonIn, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO person (name, email, phone, role, address, age) VALUES (?, ?, ?, ?, ?, ?)",
        (person.name, person.email, person.phone, person.role, person.address, person.age),
    )
    db.commit()
    new_id = cur.lastrowid
    return {**person.dict(), "id": new_id, "status": "active"}


@app.get("/api/people", response_model=List[PersonOut])
def get_people(db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT * FROM person")
    rows = cur.fetchall()
    return [{**dict(row), "status": "active"} for row in rows]


@app.get("/api/people/{person_id}", response_model=PersonOut)
def get_person(person_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT * FROM person WHERE id = ?", (person_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    return {**dict(row), "status": "active"}


@app.delete("/api/people/{person_id}", status_code=204)
def delete_person(person_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM person WHERE id = ?", (person_id,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Person not found")
    return


@app.post("/api/scans", response_model=ScanOut, status_code=201)
def create_scan(scan: ScanIn, db: sqlite3.Connection = Depends(get_db)):
    # with httpx.Client(timeout=30) as client:
    #     response = client.get("http://127.0.0.1:8999/scan")
    
    cur = db.cursor()
    enhance_fingerprint()
    emb = get_fingerprint_embedding("fingerprint.bmp", "test")
    cur.execute(
        "INSERT INTO fingerprint (person_id, finger) VALUES (?, ?)",
        (scan.person_id, scan.finger),
    )
    db.commit()
    
    new_id = cur.lastrowid
    cur.execute("INSERT INTO scans_vec VALUES (?, ?)", [new_id, serialize_float32(emb[0])] )
    db.commit()
    return FileResponse(path='fingerprint.bmp', media_type="image/bmp", filename="fingerprint.bmp")


@app.get("/api/people/{person_id}/scans", response_model=List[ScanOut])
def get_scans_by_person(person_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT * FROM fingerprint WHERE person_id = ?", (person_id,))
    rows = cur.fetchall()
    return [dict(row) for row in rows]


@app.get("/api/match")
def get_match(db: sqlite3.Connection = Depends(get_db)):
    # with httpx.Client(timeout=30) as client:
    #     response = client.get("http://127.0.0.1:8999/scan")
    enhance_fingerprint()
    emb = get_fingerprint_embedding("fingerprint.bmp", "test")
    
    cur = db.execute("SELECT p.*, (1.0 - distance) * 100 AS match_percent FROM scans_vec v JOIN fingerprint f ON v.id = f.id JOIN person p ON p.id = f.person_id WHERE v.embedding MATCH ? AND k = 5 ORDER BY v.distance",[serialize_float32(emb[0])] )
    return [dict(row) for row in cur.fetchall()]




