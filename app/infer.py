import os, glob, torch, re
from typing import List, Dict, Tuple

# ВАЖНО: в твоём model.py класс называется Net
from model import Net

LABELS = ["true","mostly-true","half-true","barely-true","false","pants-fire"]
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_-]+")

def _tokenize(t: str) -> List[str]:
    return _WORD_RE.findall(t.lower())

def _encode(text: str, word2num: Dict[str,int], max_len: int) -> List[int]:
    ids = []
    for tok in _tokenize(text):
        ids.append(word2num.get(tok, word2num.get("<unk>", 1)))
        if len(ids) >= max_len:
            break
    pad = word2num.get("<pad>", 0)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return ids

def _latest_ckpt() -> str:
    paths = sorted(glob.glob(os.path.join("models", "*.pth.tar")))
    if not paths:
        raise FileNotFoundError("В папке models нет *.pth.tar. Сначала запусти обучение (train.py).")
    return paths[-1]

class Detector:
    def __init__(self, labels=None):
        self.labels = labels or LABELS
        self.model = None
        self.word2num = None
        self.max_len = 50

    def load(self, ckpt_path: str | None = None):
        ckpt_path = ckpt_path or _latest_ckpt()
        data = torch.load(ckpt_path, map_location="cpu")
        self.word2num = data.get("word2num", {"<pad>":0, "<unk>":1})
        hyper = data.get("hyper", {})
        self.max_len = int(hyper.get("max_len", 50))

        # Размеры ставим из hyper при наличии, либо разумные дефолты
        embed_dim  = int(hyper.get("embed_dim", 100))
        num_classes = len(self.labels)

        # Конструктор Net смотри по твоему model.py (у тебя много аргументов по умолчанию)
        self.model = Net(vocab_dim=len(self.word2num), num_classes=num_classes)
        state = data["state_dict"] if "state_dict" in data else data
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        x = torch.tensor([_encode(text, self.word2num, self.max_len)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = float(probs.max().item()), int(probs.argmax().item())
        return self.labels[idx], conf
