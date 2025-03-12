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
command=[
    {"id": 1, "name": "Increase font size"},
    {"id": 2, "name": "Decrease font size"},
    {"id": 3, "name": "Increase container size"},
    {"id": 4, "name": "Decrease container size"},
    {"id": 5, "name": "Increase brightness"},
    {"id": 6, "name": "Decrease brightness"},
    {"id": 7, "name": "Navigate to home page"},
    {"id": 8, "name": "Navigate to about page"}
]


model = SentenceTransformer("all-MiniLM-L6-v2")


class Message(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"} 


@app.post("/send-message")
async def send_message(message: Message):
# print(message.message)
    test_vec = model.encode([message.message])[0]
    similarity_arr = []

    for sent in command:
        print(sent)
        
        # Calculate similarity score
        similarity_score = 1 - distance.cosine(test_vec, model.encode([sent["name"]])[0])
        
        # Convert the numpy.float32 to Python float
        similarity_score = float(similarity_score)
        
        if similarity_score > 0.01:
            similarity_arr.append({"score": similarity_score, "text": sent["name"], "id": sent["id"]})
        
    print(similarity_arr)
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    print(similarity_arr)
    return similarity_arr



# @app.post("/send-voice")
# async def send_message(message: Message):
#     test_vec = model.encode([message.message])[0]
#     similarity_arr = []
#     for sent in command:
#         similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
#         similarity_arr.append({"score": similarity_score, "text": sent})
#     similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
#     return similarity_arr[0]
