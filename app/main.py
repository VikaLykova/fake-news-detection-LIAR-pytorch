# app/main.py
from fastapi import FastAPI
from app.schemas import AnalyzeRequest, AnalyzeResponse, TrainRequest
from app.store import SessionLocal, History, init_db
import random

app = FastAPI(title="City Fake News Detector (UA)")

LABELS = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]

@app.on_event("startup")
def _startup():
    init_db()

def demo_predict(text: str):
    label = random.choice(LABELS)
    conf = round(random.uniform(0.05, 0.35), 3)
    explanation = "Демо‑API: модель будет подключена после обучения."
    return label, conf, explanation

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    label, conf, explanation = demo_predict(req.news_text)
    db = SessionLocal()
    try:
        db.add(History(text=req.news_text, label=label, confidence=conf))
        db.commit()
    finally:
        db.close()
    return AnalyzeResponse(label=label, confidence=conf, explanation=explanation)

@app.get("/history")
def history(limit: int = 20):
    db = SessionLocal()
    try:
        rows = (
            db.query(History)
              .order_by(History.id.desc())
              .limit(limit)
              .all()
        )
        return [
            {"id": r.id, "text": r.text, "label": r.label, "confidence": r.confidence}
            for r in rows
        ]
    finally:
        db.close()

@app.post("/train")
def train(_: TrainRequest):
    # заглушка — обучение подключим позже
    return {"ok": True, "message": "Training (demo) finished"}
