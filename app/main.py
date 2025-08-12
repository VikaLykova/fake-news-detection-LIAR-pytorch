from fastapi import FastAPI, HTTPException
from app.schemas import AnalyzeRequest, AnalyzeResponse, TrainRequest
from app.store import init_db, SessionLocal, History
import os, random

app = FastAPI(title="City Fake News Detector (UA)")
init_db()

LABELS = ["true","mostly-true","half-true","barely-true","false","pants-fire"]

# Пока заглушка вместо реальной модели (подключим позже).
def predict_label(text: str):
    idx = random.randint(0, len(LABELS)-1)
    return LABELS[idx], round(1.0/len(LABELS), 4)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    label, conf = predict_label(req.news_text)
    explanation = "Демо-API: модель будет подключена после обучения."
    db = SessionLocal()
    db.add(History(text=req.news_text, label=label, confidence=conf))
    db.commit()
    return AnalyzeResponse(label=label, confidence=conf, explanation=explanation)

@app.get("/history")
def history(limit: int = 20):
    db = SessionLocal()
    rows = db.query(History).order_by(History.id.desc()).limit(limit).all()
    return [{"id":r.id, "text":r.text, "label":r.label, "confidence":r.confidence} for r in rows]

@app.post("/train")
def train(req: TrainRequest):
    if not os.path.exists(req.dataset_path):
        raise HTTPException(status_code=400, detail="dataset_path not found")
    # здесь позже вызовем твой train.py и подгрузим веса
    return {"ok": True, "message": "Training (demo) finished"}
