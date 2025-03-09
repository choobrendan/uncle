from scipy.spatial import distance
from jinja2 import TemplateNotFound
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
print("YEEEEE")
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
command = [
    "Increase font size",
    "Decrease font size",
    "Increase container size",
    "Decrease container size",
    "Increase brightness",
    "Decrease brightness",
    "Navigate to home page",
    "Navigate to about page",
]

model = SentenceTransformer("all-MiniLM-L6-v2")


class Message(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"} 


@app.post("/send-message")
async def send_message(message: Message):
    print(message.message)
    test_vec = model.encode([message.message])[0]
    similarity_arr = []
    for sent in command:
        similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
        if similarity_score > 0.5:
            similarity_arr.append({"score": similarity_score, "text": sent})
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    if len(similarity_arr) == 0:
        for sent in command:
            if sent.startswith(message.message):
                similarity_arr.append(sent)
    else:
        temp = []
        for x in similarity_arr:
            temp.append(x["text"])
        similarity_arr = temp
    if len(similarity_arr) == 0:
        for sent in command:
            if message.message in sent:
                similarity_arr.append(sent)
    return similarity_arr


@app.post("/send-voice")
async def send_message(message: Message):
    test_vec = model.encode([message.message])[0]
    similarity_arr = []
    for sent in command:
        similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
        similarity_arr.append({"score": similarity_score, "text": sent})
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    return similarity_arr[0]
