from pydantic import BaseModel
from fastapi import FastAPI
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class TextIn(BaseModel):
    text: list[str]


class TextOut(BaseModel):
    sentiment: list[str]


MODEL_NAME = "Pakkapon/wangchanberta-fine-tune-fin-news-sentiment-finnlp-th"
MAX_LENGTH = 416

# ref_model_label = {"บวก": 1, "กลาง": 0, "ลบ": -1}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def sentiment_classifier(dataset):

    tokenized_texts = [
        tokenizer(
            sample,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        for sample in dataset
    ]

    with torch.no_grad():
        batch_inputs = {
            "input_ids": torch.cat([b["input_ids"] for b in tokenized_texts], dim=0),
            "attention_mask": torch.cat(
                [b["attention_mask"] for b in tokenized_texts], dim=0
            ),
        }
        logits = model(batch_inputs["input_ids"], batch_inputs["attention_mask"]).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted = probabilities.argmax(dim=1).numpy()
    result = np.array([model.config.id2label[i] for i in predicted])

    return result.tolist()


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/sentiment/")
def sentiment(full_texts: TextIn):
    full_text = full_texts.text
    sentiment = sentiment_classifier(full_text)

    return {
        "full_text": full_text,
        "sentiment": sentiment,
    }
