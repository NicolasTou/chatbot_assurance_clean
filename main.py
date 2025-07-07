from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot import get_bot_response

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def helloworld():
    return "Hello World"
@app.post("/ask")
def ask_bot(query: Query):
    return {"response": get_bot_response(query.question)}

