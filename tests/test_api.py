from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_200():
    r = client.post('/analyze', json={'news_text': 'Тестова новина'})
    assert r.status_code == 200
    body = r.json()
    assert 'label' in body and 'confidence' in body
