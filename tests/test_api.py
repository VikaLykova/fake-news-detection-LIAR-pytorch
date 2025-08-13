from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_200():
    r = client.post("/analyze", json={"news_text": "Тестовая новость"})
    assert r.status_code == 200
    body = r.json()
    assert set(["label", "confidence", "explanation"]).issubset(body.keys())
