import torch
from fake_news_detection.model import FakeNewsDetector
from fake_news_detection.utils import preprocess_text

MODEL_PATH = "models/model.pth.tar"

def load_model():
    model = FakeNewsDetector()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict(text):
    model = load_model()
    processed_text = preprocess_text(text)
    with torch.no_grad():
        output = model(processed_text)
        _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    sample_news = "Мер міста оголосив про безкоштовний проїзд у метро до кінця року"
    label = predict(sample_news)
    print(f"Predicted label: {label}")
