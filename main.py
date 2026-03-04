from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

print("Loading AI model...")

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


class Question(BaseModel):
    question: str


# ---------- WEBSITE UI ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- LOAD BREED ARTICLE ----------

def load_relevant_article(question):

    folder = "articles"
    question_lower = question.lower()

    for file in os.listdir(folder):

        breed = file.replace(".txt", "").lower()

        if breed in question_lower:
            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    text = ""

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                text += f.read()

    return text


# ---------- AI API ----------

@app.post("/ask")
def ask_ai(data: Question):

    article = load_relevant_article(data.question)

    prompt = f"""
You are a cattle expert.

Information:
{article}

Question:
{data.question}

Answer clearly:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}