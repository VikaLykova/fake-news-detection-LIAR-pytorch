from fastapi import FastAPI, HTTPException
from app.schemas import AnalyzeRequest, AnalyzeResponse, TrainRequest
from app.store import init_db, SessionLocal, History
import os, random

from app.infer import Detector, LABELS

app = FastAPI(title="City Fake News Detector (UA)")
init_db()

_detector = None

def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not os.path.isdir("models") or not any(p.endswith(".pth.tar") for p in os.listdir("models")):
        return None  # демо-режим, если нет весов
    det = Detector(labels=LABELS)
    det.load()  # возьмет самый новый чекпоинт из ./models
    _detector = det
    return _detector

def predict_label(text: str):
    try:
        det = get_detector()
        if det is None:
            raise RuntimeError("no weights")  # перейдём в демо-режим
        label, conf = det.predict(text)       # тут может упасть, если вход не того формата
        return label, round(conf, 4), "Результат получен обученной моделью."
    except Exception as e:
        # безопасный фоллбэк, чтобы не было 500
        import random
        label = random.choice(LABELS)
        conf = round(1.0/len(LABELS), 4)
        return label, conf, f"Демо-режим (fallback): {e}"


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    label, conf, explanation = predict_label(req.news_text)
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
    code = os.system("python train.py")
    if code != 0:
        raise HTTPException(status_code=500, detail="training failed")
    global _detector
    _detector = None
    return {"ok": True, "message": "Training finished. Weights in ./models/*.pth.tar"}
