import requests

BASE = "http://127.0.0.1:8000"

def test_analyze():
    r = requests.post(f"{BASE}/analyze", json={"news_text": "Тестова новина"})
    assert r.status_code == 200
    body = r.json()
    assert "label" in body and "confidence" in body and "explanation" in body

def test_history():
    r = requests.get(f"{BASE}/history?limit=1")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
