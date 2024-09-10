from typing import Union
# from fastapi import FastAPI
#update j
import os

from pydantic import BaseModel
from groq import Groq

#baru njir
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import io

from typing import List, Dict, Any

app = FastAPI()
#uvicorn main:app --reload
@app.get("/")
def read_root():
    return {"message": "Skinalyze API is running v10 sep."}


# Load models
try:
    acne_model = load_model("models/acne23augv1.h5")
    comedo_model = load_model("models/ComedoDetection_v2.h5")
    acne_level_model = load_model("models/AcneLVL_v2baru.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

@app.post("/Skinalyze-Predict/")
async def predictSkinalyze(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    IMG_WIDTH_LVL, IMG_HEIGHT_LVL = 160, 160

    try:
        # Read and compress image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Compress image by resizing and reducing quality
        img = ImageOps.exif_transpose(img)  # Handle image orientation
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_acne = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_comedo = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_acne_lvl = img.resize((IMG_WIDTH_LVL, IMG_HEIGHT_LVL), Image.LANCZOS)

        # Prepare image arrays
        img_acne_array = np.expand_dims(np.array(img_acne) / 255.0, axis=0)
        img_comedo_array = np.expand_dims(np.array(img_comedo) / 255.0, axis=0)
        img_acne_lvl_array = np.expand_dims(np.array(img_acne_lvl) / 255.0, axis=0)

        # Predict Acne
        acne_classes = acne_model.predict(img_acne_array)
        acne_class_list = ['Acne', 'Clear']
        acne_prediction = acne_class_list[np.argmax(acne_classes[0])]

        # Predict Acne Level
        if acne_prediction == 'Clear':
            acne_level_prediction = 'Level 0'
        else:
            acne_level_classes = acne_level_model.predict(img_acne_lvl_array)
            acne_level_class_list = ['Level 1', 'Level 2', 'Level 3']
            acne_level_prediction = acne_level_class_list[np.argmax(acne_level_classes[0])]

        # Predict Comedo
        comedo_classes = comedo_model.predict(img_comedo_array)
        comedo_class_list = ['Clear', 'Comedo']
        comedo_prediction = comedo_class_list[np.argmax(comedo_classes[0])]

        return JSONResponse(content={
            "acne_prediction": acne_prediction,
            "acne_level_prediction": acne_level_prediction,
            "comedo_prediction": comedo_prediction
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
#model fish or shrimp grading
#=======================================================================================================================
try:
    model = load_model("models/my_model.keras")
    print("Fish or Shrimp model loaded successfully.")
except Exception as e:
    print(f"Error loading Fish or Shrimp model: {e}")
    model = None

# Load the Fish Grading model
try:
    model_fishgrading = load_model("models/marine_grading_fish.h5")
    print("Fish grading model loaded successfully.")
except Exception as e:
    print(f"Error loading Fish grading model: {e}")
    model_fishgrading = None

# Load the Shrimp Grading model
try:
    model_shrimpgrading = load_model("models/marine_grading_shrimp.h5")
    print("Shrimp grading model loaded successfully.")
except Exception as e:
    print(f"Error loading Shrimp grading model: {e}")
    model_shrimpgrading = None

@app.post("/marine-grading/")
async def marineGrading(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    if model is None:
        raise HTTPException(status_code=500, detail="Fish or Shrimp model is not loaded")

    try:
        # Read the image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Convert to RGB if not already in that format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Compress the image by reducing its quality to 85%
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        img = Image.open(buffer)

        # Resize the image for the Fish or Shrimp model
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict Fish or Shrimp
        classes = model.predict(img_array)
        predicted_class = 'Ikan' if np.argmax(classes[0]) == 0 else 'Udang'

        # Choose the appropriate grading model and resize parameters
        if predicted_class == 'Ikan':
            if model_fishgrading is None:
                raise HTTPException(status_code=500, detail="Fish grading model is not loaded")
            grading_model = model_fishgrading
            IMG_WIDTH, IMG_HEIGHT = 160, 160
            class_list = ['A', 'B', 'C']
        else:
            if model_shrimpgrading is None:
                raise HTTPException(status_code=500, detail="Shrimp grading model is not loaded")
            grading_model = model_shrimpgrading
            IMG_WIDTH, IMG_HEIGHT = 160, 160
            class_list = ['A', 'B', 'C']

        # Resize the image for the grading model
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict the grade
        grading_classes = grading_model.predict(img_array)
        grading_result = class_list[np.argmax(grading_classes[0])]

        # Return the prediction and grading result as a JSON response
        return JSONResponse(content={"predicted_class": predicted_class, "grading_result": grading_result})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
#llammaa3-70b-8192
#=======================================================================================================================

# with open('api_key.txt', 'r') as txt_r:
#     os.environ["GROQ_API_KEY"] = txt_r.readlines()[0]

# class GroqRunTime():
#     def __init__(self):
#         self.client = Groq(
#             # this is the default and can be omitted
#             api_key=os.environ.get("GROQ_API_KEY"),
#         )

#     def generate_response(self, system_prompt, user_prompt):
#         responses = self.client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt
#                 }
#             ],
#             model = "llama3-70b-8192",
#             temperature = 0.3
#             # repetition_penalty = 0.8,
#         )
#         return responses

# class UserPrompt(BaseModel):
#     user_prompt: str

# @app.post("/chat")
# def create_chat(user_prompt: UserPrompt):
#     groq_run = GroqRunTime()

#     system_prompt = '''
#     saya ingin kamu membuat pertanyaan dari input user.
#     saya ingin kamu membuat response dalam bahasa indonesia.

#     hanya jawab dengan format dibawah ini, dan jangan ditambahkan!:

#     Pertanyaan: 
#     1. [disini kamu membuat pertanyaan pertama dari input yang diberikan oleh user].
#     2. [disini kamu membuat pertanyaan kedua dari input yang diberikan oleh user].
#     3. [disini kamu membuat pertanyaan ke-n dari input yang diberikan oleh user].
#     '''


#     # user_prompt = '''
#     # bahwa untuk melaksanakan ketentuan pasal 64 ayat (6),
#     # pasal 68 ayat (6), pasal 69 ayat (3), pasal 72 ayat (3)
#     # dan pasal 75 undang undang nomor 22 tahun 2009 tentang lalu lintas dan angkutan jalan,
#     # telah dikeluarkan  peraturan kepala kepolisian republik indonesia nomor 5 tahun 2012 tentang registrasi dan identifikasi kendaraan bermotor.
#     # '''

#     response = groq_run.generate_response(system_prompt, user_prompt.user_prompt)

#     # print(response.choices[0].message.content)
#     return {"response": response.choices[0].message.content}


# #persona chatbot 
# #=======================================================================================================================

# # Load API key
# with open('api_key.txt', 'r') as txt_r:
#     os.environ["GROQ_API_KEY"] = txt_r.readlines()[0].strip()

# class GroqRunTime2:
#     def __init__(self):
#         self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
#         self.conversations = {}  # Menyimpan riwayat percakapan berdasarkan user ID

#     def get_conversation(self, user_id):
#         return self.conversations.get(user_id, [])

#     def update_conversation(self, user_id, messages):
#         self.conversations[user_id] = messages

#     def generate_response(self, user_id, user_prompt):
#         # Ambil riwayat percakapan untuk pengguna ini
#         conversation = self.get_conversation(user_id)

#         # Tambahkan pesan pengguna ke riwayat
#         conversation.append({"role": "user", "content": user_prompt})

#         # Buat permintaan ke API dengan riwayat percakapan
#         try:
#             responses = self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "i want you to do roleplay as Keqing, you are the yuheng of liyue qixing, don't answer too long, don't answer too formally and sometimes answer a little tsundere"}
#                 ] + conversation,
#                 model="llama3-70b-8192",
#                 temperature=0.3
#                 # repetition_penalty=0.8,  # Uncomment if needed
#             )

#             # Ambil respons dari model
#             response_message = responses.choices[0].message.content  # Akses dengan atribut

#             # Tambahkan respons dari model ke riwayat
#             conversation.append({"role": "assistant", "content": response_message})

#             # Perbarui riwayat percakapan untuk pengguna ini
#             self.update_conversation(user_id, conversation)

#             return response_message
#         except Exception as e:
#             return {"error": str(e)}

# class UserPrompt2(BaseModel):
#     user_id: str
#     user_prompt: str

# @app.post("/chat")
# def create_chat2(user_prompt: UserPrompt2):
#     groq_run = GroqRunTime2()
#     response = groq_run.generate_response(user_prompt.user_id, user_prompt.user_prompt)

#     if isinstance(response, dict) and "error" in response:
#         return {"response": f"Error: {response['error']}"}

#     return {"response": response}

#persona chatbot 
#=======================================================================================================================


# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class UserPrompt(BaseModel):
#     user_id: str
#     user_prompt: str
#     previous_chat: List[ChatMessage]

# class GroqRunTime:
#     def __init__(self):
#         with open('api_key.txt', 'r') as txt_r:
#             os.environ["GROQ_API_KEY"] = txt_r.readlines()[0].strip()
#         self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#     def generate_response(self, user_id: str, user_prompt: str, previous_chat: List[Dict[str, Any]]):
#         conversation = [{"role": msg['role'], "content": msg['content']} for msg in previous_chat]
#         conversation.append({"role": "user", "content": user_prompt})

#         try:
#             responses = self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "i want you to do roleplay as Keqing, you are the yuheng of liyue qixing, don't answer too long (max 1 paragraph) and sometimes answer a little tsundere, So that you can understand the context, here is the chat history :"}
#                 ] + conversation,
#                 model="llama3-70b-8192",
#                 temperature=0.3
#             )

#             response_message = responses.choices[0].message.content

#             return response_message
#         except Exception as e:
#             return {"error": str(e)}

# @app.post("/chat-waifu")
# def create_chat(user_prompt: UserPrompt):
#     groq_run = GroqRunTime()
#     response = groq_run.generate_response(
#         user_prompt.user_id, 
#         user_prompt.user_prompt, 
#         [{"role": msg.role, "content": msg.content} for msg in user_prompt.previous_chat]
#     )

#     if isinstance(response, dict) and "error" in response:
#         return {"response": f"Error: {response['error']}"}

#     return {"response": response}


#testing
class ChatMessage(BaseModel):
    role: str
    content: str

class UserPrompt(BaseModel):
    user_id: str
    user_prompt: str
    previous_chat: List[ChatMessage]

class GroqRunTime:
    def __init__(self):
        with open('api_key.txt', 'r') as txt_r:
            os.environ["GROQ_API_KEY"] = txt_r.readlines()[0].strip()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_response(self, user_id: str, user_prompt: str, previous_chat: List[Dict[str, Any]]):
        # Menggabungkan previous_chat ke dalam user_prompt
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in previous_chat])
        full_prompt = f"chat history : {chat_history}\nuser: {user_prompt}"

        try:
            responses = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "i want you to do roleplay as Keqing, you are the yuheng of liyue qixing, don't answer too long (max 1 paragraph) and sometimes answer a little tsundere."},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3
            )

            response_message = responses.choices[0].message.content

            return response_message
        except Exception as e:
            return {"error": str(e)}

@app.post("/chat-waifu")
def create_chat(user_prompt: UserPrompt):
    groq_run = GroqRunTime()
    response = groq_run.generate_response(
        user_prompt.user_id, 
        user_prompt.user_prompt, 
        [{"role": msg.role, "content": msg.content} for msg in user_prompt.previous_chat]
    )

    if isinstance(response, dict) and "error" in response:
        return {"response": f"Error: {response['error']}"}

    return {"response": response}

