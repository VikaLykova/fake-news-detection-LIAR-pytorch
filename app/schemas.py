from pydantic import BaseModel, Field
from typing import Optional

class AnalyzeRequest(BaseModel):
    news_text: str = Field(..., description="Текст новини/заяви")
    topic: Optional[str] = None
    source: Optional[str] = None
    speaker: Optional[str] = None
    channel: Optional[str] = None  # Telegram/ЗМІ/соцмережа

class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    explanation: str

class TrainRequest(BaseModel):
    dataset_path: str
    epochs: int = 2
