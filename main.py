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
from PIL import Image
import io

from typing import List, Dict, Any

app = FastAPI()
#uvicorn main:app --reload

#load model
#=======================================================================================================================

# # Memuat model TensorFlow
# try:
#     model = load_model("models/my_model.keras")
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# # Dimensi input yang diharapkan oleh model
# IMG_WIDTH, IMG_HEIGHT = 150, 150

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model is not loaded")

#     try:
#         # Membaca file gambar
#         contents = await file.read()
#         img = Image.open(io.BytesIO(contents))

#         # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
#         if img.mode != 'RGB':
#             img = img.convert('RGB')

#         # Memproses gambar
#         img = img.resize((IMG_WIDTH, IMG_HEIGHT))

#         #array
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension
#         img_array = img_array / 255.0

#         # Melakukan prediksi
#         # predictions = model.predict(img_array)
#         # predicted_class = np.argmax(predictions, axis=1)[0]

#         #Melakukan prediksi
#         classes = model.predict(img_array, batch_size=1)
        
#         class_list = ['Ikan', 'Udang']
#         predicted_class = class_list[np.argmax(classes[0])]

#         # Mengembalikan hasil prediksi sebagai JSON response
#         # return JSONResponse(content={"predicted_class": int(predicted_class)})
#         return JSONResponse(content={"predicted_class": predicted_class})

#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")
    
#load model skin detection
#=======================================================================================================================

# Memuat model TensorFlow
try:
    model = load_model("models/AcneDetection.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/Acnepredict/")
async def predict2(file: UploadFile = File(...)):
    # Dimensi input yang diharapkan oleh model
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Membaca file gambar
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Memproses gambar
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        #array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension
        img_array = img_array / 255.0

        # Melakukan prediksi
        # predictions = model.predict(img_array)
        # predicted_class = np.argmax(predictions, axis=1)[0]

        #Melakukan prediksi
        classes = model.predict(img_array, batch_size=1)
        
        class_list = ['Clear', 'Acne']
        predicted_class = class_list[np.argmax(classes[0])]

        # Mengembalikan hasil prediksi sebagai JSON response
        # return JSONResponse(content={"predicted_class": int(predicted_class)})
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

#load model comedo detection
#=======================================================================================================================

# Memuat model TensorFlow
try:
    model = load_model("models/ComedoDetection_v2.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/Comedopredict/")
async def predict3(file: UploadFile = File(...)):
    # Dimensi input yang diharapkan oleh model
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Membaca file gambar
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Memproses gambar
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        #array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension
        img_array = img_array / 255.0

        # Melakukan prediksi
        # predictions = model.predict(img_array)
        # predicted_class = np.argmax(predictions, axis=1)[0]

        #Melakukan prediksi
        classes = model.predict(img_array, batch_size=1)
        
        class_list = ['Clear', 'Comedo']
        predicted_class = class_list[np.argmax(classes[0])]

        # Mengembalikan hasil prediksi sebagai JSON response
        # return JSONResponse(content={"predicted_class": int(predicted_class)})
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# #load model acne level
# #=======================================================================================================================

# # Memuat model TensorFlow
# try:
#     model = load_model("models/AcneLVL_v1.h5")
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None


# @app.post("/AcneLevelPredict/")
# async def predict_AcneLVL(file: UploadFile = File(...)):
#     # Dimensi input yang diharapkan oleh model
#     IMG_WIDTH, IMG_HEIGHT = 160, 160

#     if model is None:
#         raise HTTPException(status_code=500, detail="Model is not loaded")

#     try:
#         # Membaca file gambar
#         contents = await file.read()
#         img = Image.open(io.BytesIO(contents))

#         # Mengubah gambar menjadi format RGB jika tidak dalam format tersebut
#         if img.mode != 'RGB':
#             img = img.convert('RGB')

#         # Memproses gambar
#         img = img.resize((IMG_WIDTH, IMG_HEIGHT))

#         #array
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension
#         img_array = img_array / 255.0

#         # Melakukan prediksi
#         # predictions = model.predict(img_array)
#         # predicted_class = np.argmax(predictions, axis=1)[0]

#         #Melakukan prediksi
#         classes = model.predict(img_array, batch_size=1)
        
#         class_list = ['Level 0', 'Level 1', 'Level 2']
#         predicted_class = class_list[np.argmax(classes[0])]

#         # Mengembalikan hasil prediksi sebagai JSON response
#         # return JSONResponse(content={"predicted_class": int(predicted_class)})
#         return JSONResponse(content={"predicted_class": predicted_class})

#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")
    

#MODEL COMBINATION
#=======================================================================================================================
# Load Acne Detection Model
try:
    acne_model = load_model("models/AcneDetection.h5")
    print("Acne Detection Model loaded successfully.")
except Exception as e:
    print(f"Error loading Acne Detection Model: {e}")
    acne_model = None

# Load Comedo Detection Model
try:
    comedo_model = load_model("models/ComedoDetection_v2.h5")
    print("Comedo Detection Model loaded successfully.")
except Exception as e:
    print(f"Error loading Comedo Detection Model: {e}")
    comedo_model = None

# Load Acne Level Model
try:
    acne_level_model = load_model("models/AcneLVL_v1.h5")
    print("Acne Level Model loaded successfully.")
except Exception as e:
    print(f"Error loading Acne Level Model: {e}")
    acne_level_model = None

@app.post("/Skinalyze-Predict/")
async def predictSkinalyze(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    IMG_WIDTH_LVL, IMG_HEIGHT_LVL = 160, 160
    if acne_model is None or comedo_model is None or acne_level_model is None:
        raise HTTPException(status_code=500, detail="One or more models are not loaded")

    try:
        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_acne = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_comedo = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_acne_lvl = img.resize((IMG_WIDTH_LVL, IMG_HEIGHT_LVL))

        # Prepare image arrays
        img_acne_array = np.expand_dims(np.array(img_acne) / 255.0, axis=0)
        img_comedo_array = np.expand_dims(np.array(img_comedo) / 255.0, axis=0)
        img_acne_lvl_array = np.expand_dims(np.array(img_acne_lvl) / 255.0, axis=0)

        # Predict Acne
        acne_classes = acne_model.predict(img_acne_array)
        #issue njir
        acne_class_list = ['Acne', 'Clear']
        acne_prediction = acne_class_list[np.argmax(acne_classes[0])]

        if acne_prediction == 'Clear':
            # If acne prediction is clear, automatically set acne level to Level 1
            acne_level_prediction = 'Level 0'
        else:
            # If acne prediction is acne, predict the acne level
            acne_level_classes = acne_level_model.predict(img_acne_lvl_array)
            acne_level_class_list = ['Level 0', 'Level 1', 'Level 2']
            acne_level_prediction = acne_level_class_list[np.argmax(acne_level_classes[0])]

        # Predict Comedo
        comedo_classes = comedo_model.predict(img_comedo_array)
        comedo_class_list = ['Clear', 'Comedo']
        comedo_prediction = comedo_class_list[np.argmax(comedo_classes[0])]

        # Return predictions as JSON
        return JSONResponse(content={
            "acne_prediction": acne_prediction,
            "acne_level_prediction": acne_level_prediction,
            "comedo_prediction": comedo_prediction
        })

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


#persona chatbot 
#=======================================================================================================================

# Load API key
with open('api_key.txt', 'r') as txt_r:
    os.environ["GROQ_API_KEY"] = txt_r.readlines()[0].strip()

class GroqRunTime2:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversations = {}  # Menyimpan riwayat percakapan berdasarkan user ID

    def get_conversation(self, user_id):
        return self.conversations.get(user_id, [])

    def update_conversation(self, user_id, messages):
        self.conversations[user_id] = messages

    def generate_response(self, user_id, user_prompt):
        # Ambil riwayat percakapan untuk pengguna ini
        conversation = self.get_conversation(user_id)

        # Tambahkan pesan pengguna ke riwayat
        conversation.append({"role": "user", "content": user_prompt})

        # Buat permintaan ke API dengan riwayat percakapan
        try:
            responses = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "i want you to do roleplay as Keqing, you are the yuheng of liyue qixing, don't answer too long, don't answer too formally and sometimes answer a little tsundere"}
                ] + conversation,
                model="llama3-70b-8192",
                temperature=0.3
                # repetition_penalty=0.8,  # Uncomment if needed
            )

            # Ambil respons dari model
            response_message = responses.choices[0].message.content  # Akses dengan atribut

            # Tambahkan respons dari model ke riwayat
            conversation.append({"role": "assistant", "content": response_message})

            # Perbarui riwayat percakapan untuk pengguna ini
            self.update_conversation(user_id, conversation)

            return response_message
        except Exception as e:
            return {"error": str(e)}

class UserPrompt2(BaseModel):
    user_id: str
    user_prompt: str

@app.post("/chat")
def create_chat2(user_prompt: UserPrompt2):
    groq_run = GroqRunTime2()
    response = groq_run.generate_response(user_prompt.user_id, user_prompt.user_prompt)

    if isinstance(response, dict) and "error" in response:
        return {"response": f"Error: {response['error']}"}

    return {"response": response}

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


#doc dummy
#=======================================================================================================================
@app.get("/doc")
def read_doc():
    return [
    {
        "page": 0,
        "content": "PERATURAN KEPOLISlAN NEGARA REPUBLIK INDONESIA \nNOMOR 7 TAHUN 2021 \nTENTANG \nREGISTRASI DAN IDENTIFIKASI KENDARAAN BERMOTOR \nDENGAN RAHMAT TUHAN YANG MAHA ESA \nKEPALA KEPOLISlAN NEGARA REPUBLIK INDONESIA, \nMenimbang a. bahwa untuk melaksanakan ketentuan Pasal 64 ayat \n(6), Pasal 68 ayat (6), Pasal 69 ayat (3), Pasa! 72 ayat \n(3), dan Pasa! 75 Undang-Undang Nomor 22 Tahun \n2009 tentang Lalu Lintas dan Angkutan Jalan, telah \ndikeluarkan Peraturan Kepala Kepolisian Negara \nRepublik Indonesia Nomor 5 Tahun 2012 tentang \nRegistrasi dan Identiflkasi Kendaraan Bermotor; \nb. bahwa ketentuan Peraturan Kepala Kepolisian Negara \nRepublik Indonesia Nomor 5 Tahun 2012 tentang \nRegistrasi dan ldentiflkasi Kendaraan Bermotor sudah \ntidak sesuai lagi dengan perkembangan peraturan \nperundang-undangan dan kebutuhan organisasi, \nsehingga perlu diganti; \nc. bahwa berdasarkan pertimbangan sebagaimana \ndimaksud dalam huruf a dan huruf b, perlu \nmenetapkan Peraturan Kepolisian Negara Republik \nIndonesia tentang Registrasi dan IdentifIkasi \nKendaraan Bermotor; "
    },
    {
        "page": 1,
        "content": "Mengingat \nMenetapkan -2 -\n1. Undang-Undang Nomor 2 Tahun 2002 tentang \nKepolisian Negara Republik Indonesia (Lembaran \nNegara Republik Indonesia Tahun 2002 Nomor 2, \nTambahan Lembaran Negara Republik Indonesia \nNomor 4168); \n2. Undang-Undang Nomor 22 Tahun 2009 tentang Lalu \nLintas dan Angkutan Jalan (Lembaran Negara \nRepublik Indonesia Tahun 2009 Nomor 96, Tambahan \nLembaran Negara Republik Indonesia Nomor 5025); \n3. Peraturan Presiden Nomor 5 Tahun 2015 tentang \nPenyelenggaraan Sistem Administrasi Manunggal Satu \nAtap Kendaraan Bermotor (Lembaran Negara Republik \nIndonesia Tahun 2015 Nomor 6); \nMEMUTUSKAN: \nPERATURAN KEPOLISIAN NEGARA REPUBLIK INDONESIA \nTENTANG REGISTRASI DAN IDENTIFIKASI KENDARAAN \nBERMOTOR. \nBAB I \nKETENTUAN UMUM \nPasal 1 \nDalam Peraturan Kepolisian ini yang dimaksud dengan: \n1. Registrasi dan Identifikasi Kendaraan Bermotor yang \nselanjutnya disebut Regident Ranmor adalah fungsi \nkepolisian untuk memberikan legitimasi asal usul dan \nkelaikan, kepemilikan serta pengoperasian Ranmor, \nfungsi kontrol, forensik kepolisian dan pelayanan \nkepada masyarakat. \n2. Kepolisian Negara Republik Indonesia yang \nselanjutnya disebut Polri adalah alat negara yang \nberperan dalam memelihara keamanan dan ketertiban \nmasyarakat, menegakkan hukum, serta memberikan \nperlindungan, pengayoman, dan pelayanan kepada \nmasyarakat dalam rangka terpeliharanya keamanan \ndan ketertiban dalam negeri. "
    },
    {
        "page": 2,
        "content": "- 3 -\n3. Kepala Polri yang selanjutnya disebut Kapolri adalah \npimpinan Polri dan penanggung jawab \npenyelenggaraan fungsi kepolisian. \n4. Kepala Korps Lalu Lintas Polri yang selanjutnya \ndisebut Kakorlantas Polri adalah unsur pimpinan pada \nKorlantas Polri yang berkedudukan di bawah dan \nbertanggung jawab kepada Kapolri dan dalam \nmelaksanakan tugas sehari hari berada di bawah \nkendali Wakapolri. \n5. Kendaraan Bermotor yang selanjutnya disebut Ranmor \nadalah setiap kendaraan yang digerakkan oleh \nperalatan mekanik berupa mesin selain kendaraan \nyang beIjalan di atas reI. \n6. Sistem Informasi Regident Ranmor adalah \nsekumpulan subsistem yang saling berhubungan \ndengan melalui penggabungan, pemrosesan, \npenyimpanan, dan pendistribusian data yang terkait \ndengan Regident Ranmor. \n7. Nomor Registrasi Kendaraan Bermotor yang \nselanjutnya disingkat NRKB adalah tanda atau simbol \nyang berupa huruf atau angka atau kombinasi huruf \ndan angka yang memuat kode wilayah dan nomor \nregistrasi yang berfungsi sebagai identitas Ranmor. \n8. NRKB Pilihan adalah NRKB yang sudah ditetapkan \nsebagai nomor registrasi pilihan. \n9. Buku Pemilik Kendaraan Bermotor yang selanjutnya \ndisingkat BPKB adalah dokumen pemberi legitimasi \nkepemilikan Ranmor yang diterbitkan Polri dan berisi \nidentitas Ranmor dan pemilik, yang berlaku selama \nRanmor tidak dipindahtangankan. \n10. Surat Tanda Nomor Kendaraan \nselanjutnya disebut STNK adalah \nberfungsi sebagai bukti legitimasi Bermotor yang \ndokumen yang \npengoperasian \nRanmor yang berbentuk surat atau bentuk lain yang \nditerbitkan Polri yang berisi identitas pemilik, identitas \nRanmor dan masa berlaku termasuk pengesahannya. "
    },
    {
        "page": 3,
        "content": "- 4 -\n11. Tanda Nomor Kendaraan Bermotor yang seIanjutnya \ndisingkat TNKB adalah tanda Regident Ranmor yang \nberfungsi sebagai bukti legitimasi pengoperasian \nRanmor berupa pelat atau berbahan lain dengan \nspesifikasi tertentu yang diterbitkan PoIri. \n12. Surat Tanda Coba Kendaraan Bermotor yang \nselanjutnya disebut STCK adalah dokumen yang \nberfungsi sebagai bukti legitimasi pengoperasian \nsementara Ranmor sebelum diregistrasi. \n13. Tanda Coba Nomor Kendaraan Bermotor yang \nselanjutnya disebut TCKB adalah tanda yang berfungsi \nsebagai bukti legitimasi pengoperasian sementara \nRanmor sebelum diregistrasi berupa peIat atau \nberbahan lain dengan spesifikasi tertentu yang \nditerbitkan Polri. \n14. Surat Tanda Nomor Kendaraan Bermotor Lintas Batas \nNegara yang selanjutnya disebut STNK LBN adalah \ndokumen yang berfungsi sebagai bukti legalitas dan \nlegitimasi pengoperasian sementara Ranmor Asing \ndalam wilayah Negara Republik Indonesia. \n15. Tanda Nomor Kendaraan Bermotor Lintas Batas \nNegara yang selanjutnya disingkat TNKB LBN adalah \ntanda yang berfungsi sebagai bukti legitimasi \npengoperasian sementara Ranmor Asing yang \nmenggunakan STNK LBN. \n16. Bukti Regident Ranmor adalah dokumen milik negara \nyang diterbitkan setelah pelaksanaan Regident \nRanmor sebagai legitimasi kepemilikan dan \npengoperasian Ranmor. \n17. Cek Fisik Ranmor adalah proses identifikasi dan \nverifikasi Ranmor yang meliputi nomor rangka, nomor \nmesin, wama, bentuk, jenis, dan tipe Ranmor \ntermasuk pemeriksaan aspek keselamatan, \nperlengkapan, dan persyaratan telmis Ranmor untuk \nmenjamin kesesuaian antara identitas, kondisi fisik \ndengan dokumen Ranmor. "
    },
    {
        "page": 4,
        "content": "-5 -\n18. Sistem Administrasi Manunggal Satu Atap yang \nselanjutnya disebut Samsat adalah serangkaian \nkegiatan dalam penyelenggaraan Regident Ranmor, \npembayaran pajak Ranmor, bea balik nama Ranmor, \ndan pembayaran sumbangan wajib dana kecelakaan \nlalu lintas jalan secara terintegrasi dan terkoordinasi \ndalam kantor bersama Samsat. \n19. Pemblokiran \nmemberikan adalah tindakan \ntanda pada data kepolisian untuk \nRegident Ranmor \ntertentu yang merupakan pembatasan sementara \nterhadap status kepemilikan ataupun pengoperasian \nRanmor. \n20. Unit Pelaksana Regident Ranmor adalah satuan yang \nmemberikan pelayanan Regident kepemilikan \ndan/atau pengoperasian Ranmor dalam bentuk kantor \ntetap dan/ atau bergerak. \n21. Ranmor Completely Knock Down yang selanjutnya disebut \nRanmor CKD adalah Ranmor dalam keadaan terurai \nmenjadi bagian-bagian termasuk perlengkapannya serta \nmemiliki sifat utama Ranmor yang bersangkutan dan \ndirakit dan/atau diproduksi di dalam negeri. \n22. Ranmor Impor Completely Built Up yang selanjutnya \ndisebut Ranmor Impor CBU adalah impor Ranmor \ndalam keadaan jadi bagian-bagian termasuk \nperlengkapannya dalam keadaan telah terakit secara \nlengkap. \n23. Bank Persepsi adalah Bank umum yang ditunjuk oleh \nMenteri Keuangan untuk menenma setoran \npenerimaan negara. \n24. Nomor Induk Kependudukan yang selanjutnya \ndisingkat NIK adalah nomor identitas penduduk yang \nbersifat unik atau khas, tunggal dan melekat pada \nseseorang yang terdaftar sebagai penduduk Indonesia. \n25. Tanda Daftar Perusahaan yang selanjutnya disingkat \nTDP adalah surat tanda pengesahan yang diberikan \noleh kantor perusahaan kepada perusahaan "
    },
    {
        "page": 5,
        "content": "- 6 -\nperdagangan yang telah melakukan pendaftaran \nperusahaan. \n26. Surat Izin Usaha Perdagangan yang selanjutnya \ndisingkat SIUP adalah surat izin untuk dapat \nmelaksanakan kegiatan usaha perdagangan. \n27. Nomor Induk Berusaha yang selanjutnya disingkat \nNIB adalah identitas pelaku usaha yang diterbitkan \noleh Lembaga Online Single Submission. \n28. Kawasan Strategis Nasional adalah wilayah yang \npenataan ruangnya diprioritaskan karena mempunyai \npengaruh sangat penting secara nasional terhadap \nkedaulatan negara, pertahanan dan keamanan negara, \nekonomi, sosial, budaya danl atau Iingkungan \ntermasuk wilayah yang telah ditetapkan sebagai \nwarisan dunia. \n29. Tanda Bukti Pelunasan Kewajiban Pembayaran yang \nselanjutnya disingkat TBPKP adalah tanda bukti \nsetoran pelunasan kewajiban pembayaran biaya \nadministrasi STNK dan/atau TNIill, pengesahan \nSTNK, NRlill pilihan, dan Pajak Kendaraan Bermotor \n(PKB), Bea Balik Nama Kendaraan Bermotor (BBN-KB), \nserta Sumbangan Wajib Dana Kecelakaan Lalu Lintas \nJalan (SWDKLW) yang telah divalidasi. \n30. Surat Ketetapan Kewajiban \nselanjutnya disingkat SKKP Pembayaran \nadalah surat yang \nyang \ndigunakan untuk menetapkan besarnya biaya \nadministrasi STNK dan/atau TNKB, pengesahan \nSTNK, NRlill pilihan, dan besarnya Pajak Kendaraan \nBermotor (PKB), Bea Balik Nama Kendaraan Bermotor \n(BBN-KB), serta Sumbangan Wajib Dana Kecelakaan \nLalu Lintas Jalan (SWDKLW). \n31. Sertifikat Uji Tipe yang selanjutnya disingkat SUT \nadalah sertifikat sebagai bukti bahwa tipe kendaraan \nbermotor, kereta gandengan, kereta tempelan telah \nlulus uji tipe. \n32. Sertifikat Registrasi Uji Tipe yang selanjutnya \ndisingkat SRUT adalah sertifikat sebagai bukti bahwa "
    },
    {
        "page": 6,
        "content": "-7 -\nsetiap kendaraan bermotor dalam keadaan Iengkap, \nkereta gandengan, kereta ternpelan, yang dibuat \ndan/atau dirakit dan/atau diirnpor rnemiliki \nspesifikasi teknis dan unjuk kerja yang sarna/ sesuai \ndengan tipe kendaraan yang telah disahkan dan \nrnerniliki SUT. \n33. Wilayah Regident Ranrnor adalah ternpat \ndilaksanakannya Regident Ranrnor berdasarkan \ndaerah hukurn Polri. \n34. Perwakilan Negara Asing yang selanjutnya disingkat \nPNA adalah perwakilan diplornatik, dan/atau \nperwakilan konsuler yang diakreditasikan kepada \npernerintah Republik Indonesia, termasuk perwakilan \ntetap/rnisi diplornatik yang dial{reditasikan kepada \nSekretariat ASEAN, organisasi internasional yang \ndiperlakukan sebagai perwakilan diplornatik/konsuler, \nserta rnisi khusus, dan berkedudukan di Indonesia. \n35. Badan Internasional adalah suatu badan perwakilan \norganisasi internasional di bawah Perserikatan \nBangsa-Bangsa, badan-badan di bawah PNA dan \norganisasi/lembaga asing lainnya yang rneIal{sanakan \nkerja sarna teknik yang berternpat dan berkedudukan \ndi Indonesia. \n36. Konsul Kehormatan adalah warga negara Indonesia \nyang ditunjuk oleh negara asing untuk rnewakili \nkepentingan negara asing tersebut di Indonesia. \nBAB II \nPENYELENGGARAAN REGISTRASI RANMOR \nBagian Kesatu \nUrnurn \nPasa12 \n(1) Setiap Ranmor wajib diregistrasikan. \n(2) Registrasi sebagairnana dirnaksud pada ayat (1) \nrneliputi: "
    },
    {
        "page": 7,
        "content": "- 8 -\na. Registrasi Ranmor baru; \nb. Registrasi perubahan identitas Ranmor dan \npemilik; \nc. Registrasi perpanjangan Ranmor; danjatau \nd. Registrasi pengesahan Ranmor. \nPasa13 \n(1) Registrasi Ranmor dilakukan melalui Regident \nRanmor. \n(2) Regident Ranmor sebagaimana dimaksud pada ayat \n( 1) dilaksanakan pada: \na. unit pelaksana Regident Ranmor di Korlantas \nPolri; \nb. unit pelaksana Regident kepemilikan Ranmor \ndi Polda atau Polres; dan \nc. unit pelaksana Regident pengoperasian Ranmor \ndi Samsat. \nPasa14 \n(1) Registrasi Ranmor sebagaimana dimaksud dalam \nPasal 2 dilakukan terhadap Ranmor yang dimiliki: \na. perorangan; \nb. instansi pemerintah; \nc. badan usaha sesuai dengan ketentuan peraturan \nperundang-undangan; \nd. PNA; \ne. Badan Internasional; atau \nf. badan hukum asing yang berkantor tetap \ndi Indonesia. \n(2) Ranmor sebagaimana dimaksud pada ayat (1) \nmerupakan Ranmor CKD atau Ranmor Impor CBU. \nPasal5 \n(1) Ranmor yang telah diregistrasi sebagaimana dimaksud \ndalam Pasal 2, diberikan bukti registrasi Ranmor "
    },
    {
        "page": 8,
        "content": "- 9 -\nberupa: \na. BPKB; \nb. STNK; dan/atau \nc. TNKB. \n(2) Bukti registrasi Ranmor sebagaimana dimaksud pada \nayat (1) terdapat NRKB. \n(3) Pengadaan material BPKB, STNK dan TNIill \nsebagaimana dimaksud pada ayat (1) dilaksanakan \noleh Korlantas Polri. \nPasal6 \n(1) NRlill sebagaimana dimaksud dalam Pasal 5 ayat (2) \nterdiri atas: \na. kode wilayah/kode registrasi; \nb. nomor urut registrasi; dan \nc. seri huruf. \n(2) NRKB sebagaimana dimaksud pada ayat (1), ditulis \nberurutan dimulai dari kode wilayah/kode registrasi, \nnomor urut registrasi danl atau seri huruf. \n(3) Kode wilayah sebagaiman.a dimaksud pada ayat (1) \nhuruf a, diterbitkan berdasarkan wilayah registrasi \nRanmor. \n(4) Kode registrasi sebagaimana dimaksud pada ayat (1) \nhuruf a, diterbitkan berdasarkan kepentingan \npengguna Ranmor. \n(5) Nomor urut registrasi sebagaimana dimaksud pada \nayat (1) huruf b, berupa angka yang paling sedikit \nterdiri dari 1 (satu) angka dan paling banyak 4 (empat) \nangka yang ditentukan berdasarkan jenis Ranmor. \n(6) Seri huruf sebagaimana dimaksud pada ayat (1) huruf \nc, terdiri atas: \na. tanpa huruf; \nb. 1 (satu) huruf; \nc. 2 (dua) huruf; atau \nd. lebih dari 2 (dua) huruf. \n(7) Penentuan danl atau penambahan seri huruf lebih \ndari 2 (dua) huruf sebagaimana dimaksud pada ayat "
    },
    {
        "page": 9,
        "content": "-10 -\n(6) huruf d dan wilayah penggunaannya ditetapkan \ndengan Keputusan Kepala Kepolisian Daerah atas \npersetujuan Kakorlantas Polri. \n(8) Format kode wilayahjkode registrasi, nomor urut \nregistrasi, dan seri huruf sebagaimana dimaksud pada \nayat (3) sampai dengan ayat (6), tercantum dalam \nLampiran yang merupakan bagian tidak terpisahkan \ndari Peraturan Kepolisian ini. \nPasal7 \n(1) NRKB sebagaimana dimaksud dalam Pasal 6 dapat \ndimintakan NRKB pilihan dan dikenakan penerimaan \nnegara bukan pajak sesuai dengan ketentuan \nperaturan perundang- undangan. \n(2) NRKB pilihan sebagaimana dimaksud pada ayat (1), \nmeliputi pilihan nomor registrasi danjatau seri \nhurufjtanpa seri huruf. \n(3) NRKB pilihan sebagaimana dimaksud pada ayat (1), \nberlaku selama 5 (lima) tahun. \n(4) NRKB pilihan sebagaimana dimaksud pada ayat (3), \nhanya berlaku untuk satu permohonan NRKB pilihan \nyang disetujui. \n(5) NRKB yang telah disetujui sebagaimana dimaksud \npada ayat (4) tidak dapat dipindahtangankan atau \ndipindahkan ke Ranmor lain tanpa membayar \npenerimaan negara bukan pajak. \n(6) NRKB pilihan sebagaimana dimaksud pada ayat (1), \ndapat diajukan pada saat registrasi sebagaimana \ndimaksud dalam Pasal 2 ayat (2). \n(7) NRKB pilihan sebagaimana dimaksud pada ayat (1), \ndapat diterbitkan dengan tambahan persyaratan: \na. mengisi formulir permohonan; dan \nb. bukti pembayaran penerimaan negara bukan \npajak NRKB Pilihan. \n(8) Dalam hal NRKB pilihan habis masa berlaku \nsebagaimana dimaksud pada ayat (3) tidak \ndilanjutkan, Ranmor diberikan NRKB sesuai urutan. "
    },
    {
        "page": 10,
        "content": "-11 -\n(9) Penentuan NRKB Pilihan ditetapkan dengan \nKeputusan Kakorlantas Polri. \nPasal8 \nNRKB pilihan sebagaimana dimaksud dalam Pasal 7 \nditerbitkan dengan cara: \na. pemilihan dan pengecekan alokasi NRKB Pilihan oleh: \n1. petugas; atau \n2. pemohon melalui sistem NRKB Pilihan secara \nelektronik; \nb. pengajuan permohonan kepada unit pelayanan \nRegident setempat dilakukan secara manual danjatau \nelektronik; \nc. apabila NRKB pilihan dapat digunakan, pemohon \nmelakukan pembayaran penerimaan negara bukan \npajak NRKB pilihan melalui Bank Persepsi atau \nBendahara \nPenerimaan; PenerimaanjPembantu Bendahara \nd. pencetakan dan penyerahan surat keterangan NRKB \nPilihan; dan \ne. pengarsipan dokumen NRKB Pilihan. \nBagian Kedua \nRegistrasi Ranmor Baru \nPasal9 \n(1) Registrasi Ranmor baru sebagaimana dimaksud dalam \nPasa! 2 ayat (2) huruf a, dilakukan terhadap Ranmor \nyang diperoleh mela!ui: \na. hasil pembelian baru; \nb. le1ang; dan \nc. hibah. \n(2) Lelang sebagaimana dimaksud pada ayat (1) huruf b \nmeliputi: "
    },
    {
        "page": 11,
        "content": "-12 -\na. lelang penghapusan Ranmor dinas TNlfPolri; \nb. lelang temuan yang bersumber dari Direktorat \nJenderal Bea dan Cukai Kementerian Keuangan; \ndan \nc. lelang pengadilan. \n(3) Hibah sebagaimana dimal(sud pada ayat (1) huruf c, \nberupa Ranmor sebagai barang rampasan negara atau \nRanmor yang ditetapkan sebagai barang gratifikasi. \nPasallO \n(1) Registrasi Ranmor baru harus memenuhi persyaratan \npaling sedikit meliputi: \na. SRUT; \nb. bukti kepemilikan Ranmor yang sah; \nc. hasil pemeriksaan Cek Fisik Ranmor; \nd. tanda bukti identitas pemilik Ranmor; \ne. sertifikat Nomor Identifikasi Kendaraan (Vehicle \nIdentification Number) dari pabrik; dan \nf. surat kuasa jika permohonan dikuasakan oleh \npemilik Ranmor. \n(2) SRUT sebagaimana dimaksud pada ayat (1) huruf a \ndiberil<an oleh kementerian yang menyelenggarakan \nurusan pemerintahan di bidang transportasi. \n(3) Bukti kepemilikan Ranmor yang sah sebagaimana \ndimaksud pada ayat (1) hurufb, berupa: \na. faktur; \nb. kutipan risalah lelang; atau \nc. bukti hibah. \n(4) Hasil pemeriksaan Cek Fisik Ranmor sebagaimana \ndimaksud pada ayat (1) huruf c berupa: \na. formulir berita acara hasil pemeriksaan Cek Fisik \nRanmor; dan \nb. blangko Cek Fisik pemeriksaan nomor rangka \ndan nomor mesin. \n(5) Hasil pemeril<saan Cek Fisik Ranmor sebagaimana \ndimaksud pada ayat (4) dibuat sesuai standardisasi "
    },
    {
        "page": 12,
        "content": "-13 -\nspesifikasi teknis yang ditetapkan dengan Keputusan \nKakorlantas Polri. \n(6) Tanda bukti identitas pemilik Ranmor sebagaimana \ndimaksud pada ayat (1) huruf d terdiri atas: \na. untuk perseorangan, me1ampirkan: \n1. kartu tanda penduduk bagi: \na) warga negara Indonesia; atau \nb) warga negara asing yang memiliki izin \ntinggal tetap dan dilengkapi dengan \nkartu izin tinggal tetap; \n2. surat keterangan tempat tinggal bagi warga \nnegara asing yang memiliki izin tinggal \nterbatas dan dilengkapi dengan kartu izin \ntinggal terbatas; \nb. untuk badan usaha sesuai dengan ketentuan \nperaturan perundang-undangan dan badan \nhukum asing yang berkantor tetap di Indonesia, \nmelampirkan: \n1. nomor induk berusaha; \n2. nomor pokok wajib pajak; dan \n3. surat keterangan menggunakan kop surat \nbadan hukum dan ditandatangani oleh \npimpinan serta stempel/cap badan hukum \nyang bersangkutan; \nc. untuk instansi pemerintah, PNA dan Badan \nInternasional melampirkan surat keterangan \nmenggunakan Imp surat instansi yang \nditandatangani oleh pimpinan dan diberi \nstempel/cap instansi yang bersangkutan dengan \nbermeterai cukup. \nPasalll \n(1) eek Fisik sebagaimana dimal{sud dalam Pasal 10 ayat \n(4) merupakan kegiatan pemeriksaan terhadap: \na. kelengkapan dan fungsi keselamatan Ranmor; \ndan \nb. identitas Ranmor. "
    },
    {
        "page": 13,
        "content": "-14 -\n(2) Kelengkapan dan fungsi keselamatan Ranmor \nsebagaimana dimaksud pada ayat (1) huruf a, paling \nsedikit meliputi: \na. karoseri/rancang bangun sesuai peruntukan \nRanmor; \nb. lampu-lampu; \nc. kaca spion; \nd. kondisi ban; \ne. dimensi Ranmor untuk mengetahui kesesuaian \ntinggi, lebar, dan panjang; \nf. panel kontrol; dan \ng. sabuk keselamatan dan segitiga pengaman untuk \nRanmor selain jenis sepeda motor. \n(3) Identitas Ranmor sebagaimana dimaksud pada ayat (1) \nhuruf b, paling sedikit meliputi: \na. kesesuaian antara dokumen dan fisik Ranmor; \ndan \nb. hasil eek Fisik nomor rangka dan nomor mesin. \n(4) Hasil pemeriksaan eek Fisik sebagaimana dimaksud \npada ayat (2) dan ayat (3) huruf a, dicantumkan pada \nformulir berita acara hasil pemeriksaan eek Fisik \nRanmor. \n(5) Hasil pemeriksaan eek Fisik nomor rangka dan nomor \nmesin sebagaimana dimaksud pada ayat (3) huruf b, \ndicantumkan pada blangko eek Fisik pemeriksaan \nnomor rangka dan nomor mesin. \n(6) Hasil eek Fisik Ranmor sebagaimana dimaksud pada \nayat (1) dibuatkan Berita acara hasil pemeriksaan eek \nFisik Ranmor. \n(7) Berita acara hasil pemeriksaan eek Fisik Ranmor \nsebagaimana dimaksud pada ayat (6) memuat data \nRanmor, pemilik, hasil pemeriksaan, dan kesimpulan. \n(8) Berita acara hasil pemeriksaan eek Fisik Ranmor \nsebagaimana dimaksud pada ayat (7) dapat dijadikan \npertimbangan dilaksanakan atau ditolaknya Regident \nRanmor. "
    },
    {
        "page": 14,
        "content": "-15 -\n(9) Pengadaan forrnulir hasil perneriksaan Cek Fisik \nRanrnor dan blangko Cek Fisik pemeriksaan nomor \nrangka dan nomor mesin sebagaimana dimaksud pada \nayat (4) dan ayat (5) dilaksanakan secara terpusat oleh \nKorlantas Polri. \n(10) Cek Fisik Ranmor sebagaimana dimaksud pada ayat \n(1), wajib dilakukan untuk: \na. Regident Ranmor baru; \nb. Regident perubahan identitas Ranmor dan/atau \nPemilik; \nc. penggantian bukti Regident Ranmor; dan \nd. perpanjangan STNK setiap 5 (lima) tahun. \nBagian Ketiga \nRegistrasi Perubahan ldentitas Ranmor dan Pemilik \nPasal12 \nRegistrasi perubahan identitas Ranmor sebagaimana \ndimaksud dalam Pasal 2 ayat (2) huruf b, meliputi \nperubahan: \na. bentuk Ranmor; \nb. fungsi Ranmor; \nc. warna Ranmor; \nd. mesin Ranmor; dan \ne. NRKB. \nPasal13 \n(1) Regident perubahan pemilik sebagaimana dimaksud \ndalam Pasal 2 ayat (2) huruf b, meliputi perubahan: \na. nama tanpa perubahan pemilik dan alamat; \nb. alamat pemilik dan/atau nama pemilik Ranmor, \nberupa mutasi Ranmor: \n1. dalam wilayah Regident Ranmor; \n2. keluar wilayah Regident Ranmor; atau \n3. masuk wilayah Regident Ranmor. \nc. pemilik Ranmor. "
    },
    {
        "page": 15,
        "content": "-16 -\n(2) Registrasi perubahan pemilik Ranmor sebagaimana \ndimaksud pada ayat (2) dapat dilakukan karena: \n(1 ) a. jual beli; \nb. hibah; \nc. warisan; \nd. lelang; \ne. pembagian harta bersama perkawinan atas dasar \nadanya perceraian; \nf. penyertaan Ranmor sebagai modal pada badan \nusaha berbadan hukum; \ng. kepemilikan Ranmor karena adanya \npenggabungan perusahaan berbadan hukum; dan \nh. tukar-menukar. \nRegistrasi \ndimaksud Bagian Keempat \nRegistrasi Perpanjangan \nPasal 14 \nperpanjangan Ranmor sebagaimana \ndalam Pasal 2 ayat (2) huruf c, untuk \nmemperpanjang masa berlaku dengan mengganti \nSTNK dan TNKB. \n(2) Registrasi perpanjangan Ranmor sebagaimana \ndimaksud pada ayat (1) wajib diajukan permohonan \nperpanjangan sebelum masa berlaku STNK dan TNKB \nberakhir. \n(3) Registrasi perpanjangan Ranmor berfungsi sebagai \npembaruan legitimasi pengoperasian Ranmor. \nBagian Kelima \nRegistrasi Pengesahan Ranmor \nPasal15 \n(1) Registrasi pengesahan Ranmor sebagaimana dimaksud \ndalam Pasal 2 ayat (2) huruf d, berupa pengesahan \nSTNK secara berkala setiap tahun. "
    },
    {
        "page": 16,
        "content": "-17 -\n(2) Registrasi pengesahan Ranmor wajib diajukan \nsebelum masa pengesahan berakhir. \n(3) Registrasi pengesahan Ranmor berfungsi sebagai \npengawasan terhadap legitimasi pengoperasian \nRanmor. \nBAB III \nBPKB \nBagian Kesatu \nUmum \nPasal16 \n(1) BPKB sebagaimana dimal;;:sud dalam Pasal 5 ayat (1) \nhuruf a, berfungsi sebagai bukti legitimasi Ranmor \ndan kepemilikan Ranmor. \n(2) BPKB sebagaimana dimaksud pada ayat (1), paling \nsedikit memuat: \na. NRKB; \nb. nama pemilik; \nc. NIK/TDP/NIB/kartu izin tinggal tetap/kartu izin \ntinggal semen tara; \nd. alamat pemilik; \ne. nomor telepon; \nf. alamat email; \ng. merek; \nh. tipe; \n1. jenis; \nj. model; \nk. tahun pembuatan; \n1. isi sHinder / daya listrik; \nm. warna; \nn. nomor rangka; \no. nomor mesin; \np. bahan bakar / sumber energi; \nq. jumlah sumbu; \nr. jumlah roda; "
    },
    {
        "page": 17,
        "content": "-18 -\ns. nomorSRUT; \nt. nomor dokumen kepabeanan untuk Ranmor \nyang diimpor; dan \nu. nomor faktur. \n(3) BPKB berlaku selama kepemilikannya tidak \ndipindahtangankan. \n(4) BPKB sebagaimana dimaksud pada ayat (1) dibuat \nsesuai standardisasi spesifikasi teknis BPKB yang \nditetapkan dengan Keputusan Kakorlantas Polri. \n(5) BPKB sebagaimana dimaksud pada ayat (1) diterbitkan \nterhadap: \na. Ranmor baru; \nb. perubahan pemilik Ranmor; dan \nc. BPKB hilang atau rusak. \nBagian Kedua \nPersyaratan Penerbitan BPKB \nParagraf 1 \nRanmorBaru \nPasal17 \nPenerbitan BPKB baru untuk Ranmor yang diproduksi \ndan/atau dirakit dalam negeri dalam bentuk Ranmor CKD, \nharus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasall0 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. faktur Ranmor; \n4. SRUT; \n5. sertifikat Nomor Identifikasi Kendaraan dari Agen \nPemegang Merek; "
    },
    {
        "page": 18,
        "content": "-19 -\n6. rekornendasi dari instansi yang berwenang \ndi bidang penggunaan Ranrnor untuk angkutan \nurnum; dan \n7. hasil Cek Fisik Ranmor. \nPasal18 \nPenerbitan BPKB baru untuk Ra:nrnor lrnpor CBU, harus \nrnernenuhi persyaratan: \na. mengisi formulir permohonan; \nb. mela:mpirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasallO ayat (6) hurufb; \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilka:n; \n3. faktur Ra:nmor; \n4. sertifikat Nomor Identifikasi Kendaraan atau \nVehicle Identification Number, \n5. dokumen pemberitahua:n impor barang; \n6. surat keterangan imp or Ranmor yang disahka:n \npejabat bea dan cukai yang berwenang, dalam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor ta:npa pena:ngguha:n atau \npembebasa:n bea masuk; \nb) formulir B atau Otomasi data B, untuk impor \nRanmor denga:n penangguha:n bea rnasuk; \natau \nc) surat keteranga:n pemasukan Ranmor dari \nluar daerah pabea:n ke kawasa:n perdaga:nga:n \nbebas da:n pelabuhan bebas sesuai peraturan \nmenteri keuangan; \n7. SUT; \n8. SRUT; \n9. surat tanda pendaftara:n tipe untuk keperluan \nimpor dari kernenterian perindustrian; "
    },
    {
        "page": 19,
        "content": "-20-\n10. hasil penelitian keabsahan rnengenai surat \nketerangan irnpor Ranrnor yang dikeluarkan oleh \nKakorlantas Polri; \n11. surat keterangan rekondisi dari perusahaan yang \nrnerniliki izin rekondisi yang sah dilengkapi \ndengan surat izin irnpor dari kernenterian \nperdagan.gan, untuk irnpor Ranrnor bukan baru; \n12. surat izin penyelenggaraan untuk angkutan \nurnurn danl atau izin trayek dari instansi yang \nberwenang, untuk irnpor Ranrnor yang digunakan \nsebagai angkutan urnurn; dan \n13. hasil Cek Fisik Ranrnor. \nPasal19 \nPenerbitan BPKB baru untuk Ranrnor PNA, harus \nrnemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6) huruf c; \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat permohonan dari PNA; \n4. faktur Ranmor; \n5. sertifikat Nomor Identifikasi Kendaraan atau \nVehicle Identification Number, \n6. dokumen pemberitahuan impor barang, untuk \nRanmor impor CBU; \n7. surat keterangan impor Ranmor yang disahkan \npejabat Bea dan Cukai yang berwenang, dalam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor tanpa penangguhan atau pembebasan \nbea masuk; atau \nb) formulir B atau Otomasi data B, untuk impor \nRanmor dengan penangguhan bea masuk; "
    },
    {
        "page": 20,
        "content": "-21 -\n8. surat keterangan bebas pajak, untuk Ranmor \nyang diberikan fasilitas pembebasan pajak dari \nPejabat Direktorat Jendral Pajak Kementerian \nKeuangan, untuk Ranmor CKD atau Ranmor \nImpor CBU yang dibeli di dalam negeri; \n9. surat rekomendasi dan pernyataan penggunaan \nRanmor dari Kementerian Luar Negeri Republik \nIndonesia; \n10. hasil penelitian keabsahan mengenai surat \nketerangan impor Ranmor yang dikeluarkan oleh \nKakorlantas Polri; dan \n11. hasil Cek Fisik Ranmor. \nPasal20 \nPenerbitan BPKB baru untuk Ranmor Badan Internasional, \nharus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. surat permohonan dari Badan Internasional; \n2. surat kuasa bermeterai cukup, menggunakan kop \nsurat Badan Internasional dan ditandatangani \noleh pimpinan instansi yang bersangkutan; \n3. fotokopi kartu tanda penduduk yang diberi \nkuasa; \n4. surat keterangan domisili Badan Internasional; \n5. faktur Ranmor; \n6. sertifikat nomor identifikasi kendaraan atau \nvehicle identification number, \n7. dokumen pemberitahuan impor barang, untuk \nRanmorCBU; \n8. surat keterangan impor Ranmor yang disahkan \npejabat Bea dan Cukai yang berwenang, dalam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor tanpa penangguhan atau \npembebasan bea masuk; atau "
    },
    {
        "page": 21,
        "content": "-22-\nb) formulir B atau Otomasi data B, untuk impor \nRanmor dengan penangguhan bea masuk; \n9. surat keterangan bebas pajak, untuk Ranmor \nyang diberikan fasilitas pembebasan pajak dari \npejabat Ditjen Pajak Kemenkeu, untuk Ranmor \nCKD atau Ranmor Impor CBU yang dibeli \ndi dalam negeri; \n10. surat rekomendasi dan pernyataan. penggunaan \nRanmor untuk kepentingan pelaksanaan tugas \natau misi Badan Internasional dari Kementerian \nSekretariat Negara; \n11. hasH penelitian keabsahan mengenai surat \nketerangan imp or Ral1.mor yang dikeluarkan oleh \nKakorlantas Polri; dan \n12. hasH Cek Fisik Ranmor. \nPasa121 \nPenerbitan BPKB baru \npenghapusan Ranmor untuk \ndinas Ranmor hasH le1ang \nTentara Nasional \nIndonesia/Polri, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. kartu tanda penduduk; \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakHkan; \n3. surat keputusan penghapusan Ranmor dan daftar \npenghapusan Ranmor dari dinas Tentara Nasional \nIndonesia/Polri; \n4. surat penetapan pemenang dan kutipan risalah \nle1ang Ranmor; \n5. berita acara penyerahan Ranmor yang dilelang; \n6. bukti pembayaran harga le1ang; dan \n7. hasH Cek Fisik Ranmor. "
    },
    {
        "page": 22,
        "content": "-23-\nPasal22 \nPenerbitan BPKB baru untuk hasil lelang Ranmor temuan \nDirektorat Jenderal Bea dan Cukai Kementerian Keuangan, \nharus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. kartu tanda penduduk; \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat keputusan penetapan baran.g milik negara; \n4. kutipan risalah lelang Ranmor yang dibuat oleh \nbalai lelang negara; \n5. berita acara penyerahan Ranmor yang dilelang; \n6. bukti pembayaran harga lelang; \n7. SRUT; dan \n8. hasil Cek Fisik Ranmor. \nParagraf2 \nPerubahan Identitas Ranmor dan Pemilik \nPasal23 \nPerubahan data BPKB atas dasar perubahan bentuk \nRanmor, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. BPKB; \n4. STNK; \n5. rekomendasi dari unit pelaksana Regident untuk \nrubah bentuk; \n6. surat keterangan dari Agen Pemegang Merek atau \nbengkel umum yang melaksanakan perubahan "
    },
    {
        "page": 23,
        "content": "-24-\nbentuk Ranmor yang disertai TDP/NIB, SIUP, \nnomor pokok wajib pajak dan surat keterangan \ndomisili; dan \n7. hasil Cek Fisik Ranmor. \nPasa124 \nPerubahan data BPKB atas dasar perubahan fUngsi \nRanmor, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6) huruf a dan huruf b; \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. BPKB; \n4. STNK; \n5. surat izin penyelenggaraan angkutan umum dari \ninstansi yang berwenang, untuk perubahan \nfungsi dari Ranmor perseorangan menjadi \nRanmor angkutan umum; \n6. surat keterangan dari instansi yang \nberwenang,untuk perubahan fungsi dari Ranmor \nangkutan umum menjadi Ranmor perseorangan; \ndan \n7. hasil Cek Fisik Ranmor. \nPasa125 \nPerubahan data BPKB atas dasar perubahan warna \nRanmor, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \npada Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; "
    },
    {
        "page": 24,
        "content": "-25 -\n3. BPKB; \n4. STNK; \n5. rekomendasi dari unit pelaksana Regident untuk \nperubahan warna Ranmor; \n6. surat keterangan dari bengkel umum yang \nmelaksanakan perubahan warna Ranmor yang \ndisertai TDP/NIB, SIUP, nomor pokok wajib pajak \ndan surat keterangan domisili; dan \n7. hasil Cek Fisik Ranmor. \nPasal26 \n(1) Perubahan data BPKB atas dasar perubahan mesin \nbaru, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nkartu tanda penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. BPKB; \n4. STNK; \n5. rekomendasi dari unit pelal<sana Regident \nuntuk ganti mesin baru; \n6. faktur pembelian mesin dari agen pemegang \nmerek; \n7. dokumen pemberitahuan impor barang; \n8. surat keterangan dari bengkel resmi agen \npemegang merek yang melaksanakan \npenggantian mesin yang disertai TDP /NIB, \nSIUP, Nomor Pokok Wajib Pajak dan surat \nketerangan domisili; \n9. hasil Cek Fisik Ranmor. \n(2) Perubahan data BPKB atas dasar perubahan mesm \nbukan baru dari Ranmor lain, harus memenuhi \npersyaratan: "
    },
    {
        "page": 25,
        "content": "-26-\na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cuku p dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. BPKB; \n4. STNK; \n5. rekomendasi dari unit pelaksana Regident \nuntuk ganti mesin bukan baru; \n6. surat keterangan dari bengkel resmi agen \npemegang merek atau bengkel umum yang \nmelaksanakan penggantian mesin yang \ndisertai TDP/NIB, SIUP, Nomor Pokok Wajib \nPajak dan surat keterangan domisili; \n7. BPKB dan STNK dari asal usul mesin \npengganti; dan \n8. hasil Cek Fisik Ranmor. \nPasal 27 \nPerubahan data BPKB atas dasar perubahan NRKB, harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. BPKB; \n4. STNK; \n5. tanda bukti pembayaran penerimaan negara \nbukan pajak dan surat keterangan untuk NRKB \npilihan; dan \n6. hasil Cek Fisik Ranmor. "
    },
    {
        "page": 26,
        "content": "-27 -\nPasa128 \n(1) Perubahan data BPKB atas dasar perubahan nama \ntanpa perubahan pemilik dan alamat, harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu TaJ.1.da Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. al(ta perubahan nama bagi badan hukum; \n4. penetapan pengadilan bagi pemilik \nperorangan; \n5. BPKB; \n6. STNK; dan \n7. hasil Cek Fisik Ranmor. \n(2) Perubahan data BPKB atas dasar perubahan alamat \npemilik Ranmor dalam satu wilayah Regident, harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. akta perubahan alamat bagi badan hukum; \n4. BPKB; \n5. STNK; dan \n6. hasil Cek Fisik Ranmor. \n(3) Perubahan data BPKB atas dasar perubahan alamat \npemilik ke luar wilayah Regident, harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: "
    },
    {
        "page": 27,
        "content": "-28 -\n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasallO ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkru:l; \n3. akta perubahan alamat bagi badan hukum. \n4. BPKB; \n5. STNK; \n6. tanda bukti pembayaran penerimaan negara \nbukan pajak Mutasi Ranmor keluar daerah; \ndan \n7. hasil Cek Fisik Ranmor. \nPasal29 \nDalam hal terdapat perubahan pemilik Ranmor dilakukan \npenggantian BPKB, dengan persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. BPKB; \n4. STNK; \n5. bukti pemindahtanganan kepemilikan; \n6. surat pengantar Mutasi \nperubahan pemilik keluar \nRanmor; Ranmor untuk \nwilayah Regident \n7. tanda bukti pembayaran penerimaan negara \nbukan pajak;dan \n8. hasil Cek Fisik Ranmor. \nPasa130 \nBukti pemindahtanganan kepemilikan sebagaimana \ndimaksud dalam Pasal 29 huruf b angka 5, dapat berupa: "
    },
    {
        "page": 28,
        "content": "-29 -\na. kuitansi pembelian bermeterai cukup dan/atau surat \npe1epasan hak, bagi pemindahtanganan karena jual \nbeli; \nb. akta waris bagi pemindahtanganan karena warisan; \nc. kutipan risalah lelang untuk: \n1. Ranmor hasil lelang penghapusan dinas instansi \npemerintah; \n2. Ranmor hasil lelang sebagai barang \nnegara yang dilengkapi dengan \npengadilan yang telah mempunyai \nhukum tetap; dan rampasan \nputusan \nkekuatan \n3. Ranmor hasille1ang sebagai barang gratifikasi; \nd. akta hibah bagi pemindahtanganan karena hibah; \ne. akta penyertaan bagi pemindahtanganan karena \npenyertaan Ranmor sebagai modal; \nf. al<ta penggabungan bagi pemindahtanganan karena \npenggabungan perusahaan berbadan hukum; \ng. akta pembagian harta benda bagi pemindahtanganan \nkarena perceraian; atau \nh. akta pernyataan tukar menukar dari kedua belah \npihak; \n1. surat rekomendasi dari: \n1. Kementerian Luar Negeri Republik Indonesia \nuntuk Ranmor PNA; atau \n2. Kementerian Sekretariat Negara Republik \nIndonesia untuk Ranmor Badan Internasional. \nPasal31 \nDalam hal terdapat perubahan pemilik Ranmor untuk \nRanmor bekas Badan Internasional dan PNA dilakukan \npenggantian BPKB, dengan persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasallO ayat (6); "
    },
    {
        "page": 29,
        "content": "-30-\n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat permohonan dari Badan Internasional atau \nPNA; \n4. kuitansi pembelian bermeterai cukup; \n5. surat keterangan pelepasan hak dari Badan \nInternasional atau PNA yang bersangkutan; \n6. BPKB; \n7. STNK; \n8. dokumen kepabeanan, meliputi: \na) formulir B atau Otomasi data B, sebagai tanda \nbukti penangguhan bea masuk atau surat \nketerangan bebas pajak pertambahan nilai \natau pajak pertambahan nilai dan pajak \npenjualan atas barang mewah, untuk \npemindahtanganan kepemilikan antar Badan \nInternasional atau PNA; atau \nb) formulir C atau Otomasi data C sebagai \ntanda bukti pelunasan bea masuk, untuk \npemindahtanganan kepemilikan kepada \nperorangan/badan hukum; \n9. rekomendasi dari Kementerian Luar Negeri untuk \nRanmor PNA atau surat rekomendasi Sekretariat \nNegara untuk Ranmor Badan Internasional; \n10. hasil penelitian keabsahan mengenai surat \nketerangan impor Ranmor yang dikeluarkan oleh \nKakorlantas Polri; \nII. Surat pengantar mutasi Ranmor untuk \nperubahan pemilik keluar wilayah Regident \nRanmor; \n12. tanda bukti pembayaran penerimaan negara \nbukan pajak; dan \n13. hasil Cek Fisik Ranmor. "
    },
    {
        "page": 30,
        "content": "-31 -\nParagraf3 \nBPKB Hilang atau Rusak \nPasal32 \n(1) Dalam hal BPKB hilang atau rusak, pemilik Ranmor \ndapat mengajukan permohonan penggantian. \n(2) Penggantian BPKB karen a hilang dilaksanakan dengan \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasall0 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. Surat Tanda Penerimaan Laporan dari Polri; \n4. Berita Acara PemeriksaaIl dari Penyidik Polri; \n5. STNK; \n6. tanda bukti pembayaran penerimaan negara \nbukan pajak; \n7. surat pernyataan pemilik bermeterai cukup \nmengenai BPKB yang hilang tidak terkait \nkasus pidana dan perdata; \n8. bukti pengumuman pada media cetak \nsebanyak 3 (tiga) kali berturut-turut dengan \ntenggang waktu setiap bulan 1 (satu) kali, \nbulan pertama media cetak lokal, bulan \nkedua dan bulan ketiga pada media cetak \nnasional; dan \n9. hasil eek Fisik Ranmor. \n(3) Penggantian BPKB karena rusak harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); "
    },
    {
        "page": 31,
        "content": "-32 -\n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. BPKB yang rusak; \n4. tanda bukti pembayararl penerimaan negara \nbukan pajak; \n5. STNK; dan \n6. hasil eek Fisik Ranmor. \nBagian Ketiga \nProsedur Penerbitan BPKB \nPasal33 \n(1) Penerbitan BPKB dilalmkan dengan pengajuan \npermohonan secara tertulis oleh pemohon kepada Unit \nPelaksana Regident Ranmor. \n(2) Terhadap permohonan sebagaimana dimaksud pada \nayat (1) dilaksanal(an tahapan kegiatan: \na. identifikasi dan verifikasi; \nb. pembayaran Penerimaan Negara Bukan Pajak; \nc. pendaftaran; \nd. pencetakan; \ne. penyerahan; dan \nf. pengarslpan. \n(3) Tahapan kegiatan sebagaimana dimaksud ayat (2) \ndilakukan oleh kelompok kerja. \nPasa134 \n(1) Identifikasi dan verifilmsi sebagaimana dimaksud \ndalam Pasal 33 ayat (2) huruf a, merupakan kegiatan \npemeriksaan berkas permohonan registrasi Ranmor \nbaru untuk penerbitan BPKB. \n(2) Kegiatan pemeriksaan berkas perrnohonan \nsebagaimana dimaksud pada ayat (1) paling sedikit \nmeliputi: \na. penelitian kelengkapan dokumen persyaratan; \nb. pemeriksaan keabsahan dokumen persyaratan; "
    },
    {
        "page": 32,
        "content": "-33 -\nc. pencocokan hasil pemeriksaan eek Fisik Ranmor; \nd. pemeriksaan kesesuaian antara dokumen \nasal-usul, kelaikan dan kepemilikan Ranmor; dan \ne. pemilahan dokumen persyaratan untuk proses \npenerbitan: \n1. BPKB, untuk diserahkan kepada kelompok \nkerja pendaftaran BPKB; dan \n2. STNK, untuk diserahkan kepada kelompok \nkerja pendaftaran STNK. \n(3) Terhadap berkas permohonan sebagaimana dimaksud \npada ayat (1) tidak lengkap, diberitahukan secara \ntertulis dan dikembalikan kepada pemohon untuk \nmelengkapi persyaratan. \n(4) Terhadap berkas permohonan sebagaimana dimaksud \npada ayat (1) telah dinyatakan sesuai, pemohon \nregistrasi Ranmor diberikan surat ketetapan kewajiban \npembayaran Penerimaan Negara Bukan Pajak. \nPasal 35 \n(1) Pembayaran Penerimaan Negara Bukan Pajak \nsebagaimana dimaksud dalam Pasal 33 ayat (2) huruf \nb, dilakukan oleh pemohon registrasi Ranmor melalui \nBank Persepsi secara tunai atau non tunai sesuai \ndengan ketentuan peraturan perundang-undangan. \n(2) Dalam hal tidak terdapat Bank Persepsi sebagaimana \ndimaksud pada ayat (1) pembayaran penerimaan \nnegara bukan pajak dilakukan melalui Bendahara \nPenerimaanjPembantu Bendahara Penerimaan sesuai \ndengan ketentuan peraturan perundang-undangan. \n(3) Pemohon registrasi Ranmor yang telah melakukan \npembayaran Penerimaan Negara Bukan Pajak \nsebagaimana dimaksud pada ayat (1) dan ayat (2) \ndiberikan bukti pembayaran penerimaan negara \nbukan pajalc "
    },
    {
        "page": 33,
        "content": "-34-\nPasal36 \n(1) Pendaftaran sebagaimana dimaksud dalam Pasal 33 \nayat (2) huruf c, merupakan kegiatan mendaftarkan \nRanmor baru yang telah memenuhi persyaratan dan \ntelah membayar Penerimaan Negara Bukan Pajak ke \ndalam buku register danl atau secara elektronik pada \nsistem manajemen registrasi Ranmor. \n(2) Kegiatan mendaftarkan Ranmor baru sebagaimana \ndimaksud pada ayat (1) paling sedikit meliputi: \na. mencatat atau memasukkan data identitas \npemilik, identitas Ranmor dan asal-usul Ranmor; \nb. penetapan nomor seri BPKB; \nc. penetapan NRKB; \nd. pencetakan kartu Induk BPKB; \ne. penandatanganan Kartu Induk BPKB oleh \npetugas verifikator; \nf. penyerahan dokumen persyaratan dan Kartu \nInduk BPKB kepada Kelompok Kerja pencetakan \ndan penyerahan;dan \ng. pemberian tanda bukti pendaftaran BPKB dan \ndokumen persyaratan penerbitan STNK kepada \npemohon sebagai dasar untuk mengajukan \npermohonan penerbitan STNK di Samsat. \n(3) Terhadap pendaftaran sebagaimana dimaksud pada \nayat (2) Ranmor diberikan NRKB, nomor seri BPKB \ndan Kartu Induk BPKB. \nPasal37 \n(1) Pencetakan sebagaimana dimaksud dalam Pasal 33 \nayat (2) huruf d, merupakan kegiatan mencetal{ \nidentitas pemilik, identitas kendaraan, dan data \npersyaratan registrasi pertama pada BPKB. \n(2) Sebelum melalmkan pencetakan sebagaimana \ndimaksud pada ayat (1), kelompok kerja pencetakan \nmelakukan: "
    },
    {
        "page": 34,
        "content": "-35-\na. verifikasi kesesuaian data antara dokumen \npersyaratan BPKB dengan data sistem informasi \nregistrasi Ranmor; dan \nb. memasukkan tanda tangan elektronik pejabat \nyang berwenang. \n(3) Pejabat yang berwenang sebagaimana dimaksud pada \nayat (2) huruf b, meliputi: \na. Direktur Lalu Lintas Kepolisian Daerah, untuk \nBPKB yang diterbitkan Direktorat Lalu Lintas \nKepolisian Daerah; atau \nb. Kepala Kepolisian Resor, untuk BPKB yang \nditerbitkan Kepolisian Resor. \nPasal38 \n(1) Penyerahan sebagaimana dimaksud dalam Pasal 33 \nayat (2) huruf e, merupakan kegiatan penyerahan \nBPKB kepada pemohon registrasi disertai dengan \ntanda bukti pendaftaran BPKB. \n(2) Tanda bukti pendaftaran BPKB sebagaimana \ndimaksud pada ayat (1) diberikan sebagai syarat \npermohonan penerbitan STNK dan TNKB. \n(3) Penyerahan sebagaimana dimaksud pada ayat (1) \ndilakukan setelah pemohon menandatangani buku \nregister penyerahan BPKB. \nPasal 39 \nPengarsipan sebagaimana dimal{sud dalam Pasal 33 ayat \n(2) huruf f, merupakan kegiatan: \na. pencatatan dan pendataan dokumen persyaratan \npenerbitan BPKB pada buku register pengarsipan atau \nsistem manajemen registrasi Ranmor; \nb. pengelompokan dokumen persyaratan penerbitan \nBPKB menurut nomor sen BPKB dan jenis Ranmor \natau NRKB; dan \nc. penataan dan penyimpanan arSlp secara manual \ndan I atau elektronik sesuai dengan ketentuan \nperaturan perundang-undangan. "
    },
    {
        "page": 35,
        "content": "-36-\nPasa140 \n(1) Dalam hal mutasi Ranmor keluar wilayah Regident \nRanmor sebagaimana dimaksud dalam Pasal 13 ayat \n(1) huruf b angka 2, dilaksanakan oleh kelompok kerja \npada unit pelaksana Regident mutasi Ranmor. \n(2) Kelompok kerja pada unit pelaksana Regident Mutasi \nRanmor sebagaimana dimaksud pada ayat (1) \nmelakukan kegiatan: \na. pemeriksaan berkas permohonan sebagaimana \ndimaksud dalam Pasal34 ayat (2); \nb. pemberitahuan secara tertulis dan mengembalikan \nkepada pemohon untuk melengkapi persyaratan, \napabila dokumen persyaratan sebagaimana dimaksud \npada huruf a tidal( lengkap; \nc. pemberitahuan kepada pemohon untuk melalrukan \npembayaran penerimaan negara bukan pajak mutasi \nkeluar wilayah Regident Ranmor; \nd. penerimaan bukti pembayaran penerimaan negara \nbukan pajak mutasi keluar wilayah Regident \nRanmor; \ne. pencetakan tujuan mutasi keluar wilayah \nRegident Ranmor pada lembar perubahan \ndi BPKB; \nf. pengambilan arsip BPKB pada unit pelaksana \nRegident kepemilikan Ranmor dan arsip STNK \npada unit pelaksana regident pengoperasian \nRanmor; \ng. penggabungan arsip BPKB dan STNK; \nh. pencetakan surat pengantar mutasi keluar \nwilayah Regident Ranmor; \ni. penandatanganan dokumen mutasi keluar wilayah \nRegident Ranmor oleh pejabat yang berwenang \nsecara manual dan/atau elektron&; \nj. penyerahan dokumen mutasi keluar wilayah \nRegident Ranmor dan surat keterangan pengganti \nSTNK kepada pemohon untuk diserahkan kepada \nunit pelaksana Regident mutasi Ranmor tujuan; "
    },
    {
        "page": 36,
        "content": "-37 -\nk. pencatatan dalarn buku register dan pangkalan \ndata sistem informasi Regident Ranmor; dan \n1. penghapusan data Regident Ranmor pada unit \nlayanan BPKB dan Sarnsat, setelah dilakukan \npengecekan silang data Ranmor secara elektronik \ndanl atau melalui surat dari unit pelal{sana \nRegident mutasi Ranmor tujuan. \n(3) Surat l<:eterangan pengganti STNK sebagaimana \ndimaksud pada ayat (2) huruf 1, berfungsi sebagai \npengganti STNK selarna proses mutasi keluar wilayah \nRegident Ranmor, yang berlaku paling lama \n60 (enam puluh) hari. \nPasa141 \n(1) Berdasarkan penyerahan dokumen Mutasi Ranmor \nkeluar wilayah Regident sebagaimana dimaksud dalam \nPasal40 ayat (2) hurufj, pada unit pelaksana Regident \nmutasi Ranmor Direktorat Lalu Lintas Kepolisian \nDaerah/Kepolisian Resor tujuan, petugas melakukan: \na. pemberitahuan kepada pemohon untuk \nmelakukan Cek Fisik Ranmor; \nb. pencocokan hasil Cek Fisik Ranmor dengan \ndokumen mutasi Ranmor keluar wilayah Regident \nsebagaimana dimaksud dalam Pasal 40 ayat (2) \nhurufj; \nc. penelitian kelengkapan, keabsahan dan \npengecekan silang dokumen mutasi Ranmor \nkeluar wilayah Regident sebagaimana dimaksud \ndalam Pasal 40 ayat (2) huruf j, ke unit pelaksana \nRegident mutasi Ranmor asal secara manual \ndanl atau elektronik; \nd. pemberitahuan secara tertulis dan \nmengembalikan kepada pemohon untuk \nmelengkapi persyaratan, apabila hasil penelitian \nsebagaimana dimaksud pada huruf c, dokumen \ntidak lengkap; "
    },
    {
        "page": 37,
        "content": "-38-\ne. pemilahan dokumen mutasi Ranmor keluar \nwilayah Regident sebagaimana dimaksud dalam \nPasal40 ayat (2) hurufj, menjadi 2 bagian yaitu: \n1. arsip BPKB; dan \n2. arsip STNK; \nf. penyerahan arsip BPKB sebagaimana dimaksud \npada huruf e angka 1 kepada unit pelaksana \nRegident kepemilikan Ranmor untuk penerbitan \nBPKB;dan \ng. penyerahan arsip STNK sebagaimana dimaksud \npada huruf e angka 2 kepada unit pelaksana \nRegident pengoperasian Ranmor untuk \npenerbitan STNK dan TNKB dengan melampirkan \ntanda bukti pendaftaran BPKB. \nPasal42 \n(1) Setiap kelompok kerja pada Unit Pelaksana Regident \nmutasi Ranmor sebagaimana dimaksud dalam Pasal \n40 ayat (1), wajib mencatat semua kegiatan dan \nkejadian dalam buku register dan/ atau secara \nelektronik pada pangkalan data sistem informasi \nRegident Ranmor. \n(2) Unit pelaksana Regident Mutasi Ranmor sebagaimana \ndimaksud pada ayat (1), berkedudukan di: \na. kantor Direktorat Lalu Lintas Kepolisian Daerah; \nb. Satuan Lalu Lintas Kepolisian Resor; atau \nc. kantor bersama Samsat. \nBABrv \nSTNK DAN TNKB \nBagian Kesatu \nUmum \nPasal43 \n(1) STNK sebagaimana dimaksud dalam Pasal 5 ayat (1) \nhuruf b, paling sedikit memuat: "
    },
    {
        "page": 38,
        "content": "-39-\na. NRKB; \nb. nama pemiIik; \nc. NIKjTDPjNIBjkartu izin tinggal tetapjkartu izin \ntinggal semen tara; \nd. alamat pemilik; \ne. merek; \nf. tipe; \ng. jenis; \nh. model; \ni. tahun pembuatan; \nj. isi silinderjdaya listrik; \nk. warna; \n1. nomor rangka; \nm. nomor mesin; \nn. nomor BPKB; \no. masa berlaku; \np. wama TNKB; \nq. tahun registrasi; \nr. bahan bakar j sumber energi; \ns. kode lokasi; dan \nt. nomor urut register. \n(2). STNK berfungsi sebagai bukti legitimasi pengoperasian \nRanmor. \n(3) STNK berlaku selama 5 (lima) tahun sejak tanggal \nditerbitkan. \n(4) Dalam hal penerbitan STNK terhadap perubahan \nidentitas Ranmor, perubahan alamat pemilik Ranmor \ndalam wilayah Regident yang sarna dan STNK \nhilangjrusak, masa berlaku STNK melanjutkan masa \nberlaku sebelumnya. \n(5) STNK beserta komponen pendukungnya menggunakan \nstandardisasi spesifikasi teknis material yang \nditetapkan dengan Keputusan Kakorlantas Polri. \n(6) Pengadaan material STNK dan komponen \npendukungnya diselenggarakan secara terpusat oleh \nKorlantas Polri. "
    },
    {
        "page": 39,
        "content": "-40-\nPasa144 \n(1) TNKB sebagaimana dimaksud dalam Pasal 5 ayat (1) \nhuruf c, memuat: \na. NRKB; dan \nb. masa berlaku. \n(2) Masa berlaku TNKB harus sesuai dengan masa \nberlaku STNK. \nPasal45 \n(1) TNKB sebagaimana dimaksud dalam Pasal 44 ayat (1) \nberwarna dasar: \na. putih, tulisan hitam untuk Ranmor perseorangan, \nbadan hukum, PNA dan Badan Internasional; \nb. kuning, tulisan hitam untuk Ranmor umum; \nc. merah, tulisan putih untuk Ranmor instansi \npemerintah; dan \nd. hijau, tulisan hitam untuk Ranmor di kawasan \nperdagangan bebas yang mendapatkan fasilitas \npembebasan bea masuk dan berdasarkan \nketentuan peraturan perundang-undangan. \n(2) Warna TNKB sebagaimana dimaksud pada ayat (1) \nditambahkan tanda khusus untuk Ranmor listrik yang \nditetapkan dengan Keputusan Kakorlantas Polri. \n(3) TNKB dipasang pada tempat yang disedial<:an dibagian \ndepan dan belakang Ranmor yang mudah terlihat dan \nteridentifikasi. \n(4) Standardisasi spesifikasi telmis TNKB ditetapkan \ndengan Keputusan Kalmrlantas Polri. \n(5) Pengadaan material TNKB diselenggarakan secara \nterpusat oleh Korlantas Polri. "
    },
    {
        "page": 40,
        "content": "-41 -\nBagian Kedua \nPersyaratan STNK \nParagraf 1 \nPenerbitan STNK Baru \nPasal46 \nPenerbitan STNK baru untuk Ranmor CKD, memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. me1ampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. faktur Ranmor; \n4. sertifikat Nomor Identifikasi Kendaraan dari Agen \nPemegang Merek; \n5. rekomendasi dari instansi yang berwenang \ndi bidang penggunaan Ranmor untuk angkutan \numum; \n6. hasil Cek Fisik Ranmor; dan \n7. tanda bukti pendaftaran BPKB. \nPasal47 \nPenerbitan STNK baru untuk Ranmor unpor CBU, harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6) huruf b; \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang dibeli kuasa bagi yang \ndiwakilkan; \n3. faktur Ranmor; "
    },
    {
        "page": 41,
        "content": "-42-\n4. sertifikat Nomor Identifikasi Kendaraan atau \nVehicle Identification Number, \n5. dokumen pemberitahuan impor barang; \n6. surat keterangan impor Ranmor yang disahkan \npejabat Bea dan Cukai yang berwenang, dalam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor tanpa penangguhan atau \npembebasan bea masuk; \nb) formulir B atau Otomasi data B, untuk impor \nRanmor dengan penangguhan bea masuk; \natau \nc) surat keterangan pemasukan Ranmor dari \nluar daerah pabean ke kawasan perdagangan \nbebas dan pelabuhan bebas sesuai peraturan \nmenteri keuangan; \n7. SUT; \n8. SRUT; \n9. surat tanda pendaftaran tipe untuk keperluan \nimpor dari kementerian perindustrian; \n10. hasil penelitian keabsahan mengenaI surat \nketerangan impor Ranmor yang dikeluarkan oleh \nKakorlantas Polri; \n11. surat keterangan rekondisi dari perusahaan yang \nmemiliki izin rekondisi yang sah dilengkapi \ndengan surat izin impor dari kementerian \nperdagangan, untuk impor Ranmor bukan baru; \n12. surat izin penyelenggaraan untuk angkutan \numum dan/atau izin trayek dari instansi yang \nberwenang, untuk impor Ranmor yang digunakan \nsebagai angkutan umum; \n13. hasil Cek Fisik Ranmor; dan \n14. tanda bukti pendaftaran BPKB. \nPasal48 \nPenerbitan STNK baru untuk RaI~mor PNA harus \nmemenuhi persyaratan: "
    },
    {
        "page": 42,
        "content": "-43-\na. ITlengisi forITlulir per=ohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasallO ayat (6) huruf c; \n2. surat kuasa ber=eterai cukup dan fotokopi kartu \ntanda penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat permohonan dari PNA; \n4. faktur Ranmor; \n5. sertifikat Nomor Identifikasi Kendaraan atau \nVehicle Identification Number, \n6. dokumen pemberitahuan impor barang, untuk \nRanmor imp or CBU; \n7. surat keterangan impor Ranmor yang disahkan \npejabat Bea dan Cukai yang berwenang, dalam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor tanpa penangguhan atau \npembebasan bea masuk; atau \nb) for=ulir B atau Otomasi data B, untuk impor \nRanmor dengan penangguhan bea masuk; \n8. surat Keterangan Bebas Pajak, untuk Ranmor \nyang diberikan fasilitas pembebasan pajak dari \npejabat Di1jen Pajak Kemenkeu, untuk Ranmor \nCKD atau Ranmor Impor CBU yang dibeli \ndi dalam negeri; \n9. surat rekomendasi dan pemyataan penggunaan \nRanmor dari Kementerian Luar Negeri; \n10. hasH penelitian keabsahan mengenm surat \nketerangan impor Ranmor yang dikeluarkan oleh \nKakorlantas Polri; \n11. hasH Cek Fisik Ranmor; dan \n12. tanda bukti pendaftaran BPKB. \nPasa149 \nPenerbitan STNK baru untuk Ranmor Badan Intemasional, \nharus memenuhi persyaratan: "
    },
    {
        "page": 43,
        "content": "-44-\na. rnengisi forrnulir perrnohonan; \nb. meIampirkan: \n1. surat permohonan dari Badan InternasionaI; \n2. surat kuasa bermeterai cukup, menggunakan kop \nsurat Badan Internasional dan ditandatangani \noIeh pimpinan instansi yang bersangkutan; \n3. surat keterangan domisili Badan Internasional; \n4. fotokopi kartu tanda penduduk yang diberi \nkuasa; \n5. faktur Ranmor; \n6. Sertifikat Nomor Identifikasi Kendaraan atau \nVehicle Identification Number, \n7. dokumen pemberitahuan impor barang, untuk \nRanmorCBU; \n8. surat keterangan impor Ranmor yang disahkan \npejabat Bea dan Cukai yang berwenang, daIam \nbentuk: \na) formulir A atau Otomasi data A, untuk impor \nRanmor tanpa penangguhan atau \npembebasan bea masuk; atau \nb) formulir B atau Otomasi data B, untuk impor \nRanmor dengan penangguhan bea masuk; \n9. surat keterangan bebas pajak, untuk Ran.mor \nyan.g diberikan fasilitas pembebasan pajak dari \npejabat Ditjen Pajak Kemenkeu, untuk Ranmor \nCKD atau Ranmor Impor CBU yang dibeli \ndi dalam negeri; \n10. surat rekomendasi dan pernyataan penggunaan \nRanmor untuk kepentingan peIaksanaan tugas \natau misi Badan Internasional dari Kementerian \nSekretariat Negara; \n11. hasil penelitian keabsahan mengenaJ. surat \nketerangan impor Ranmor yang dikeluarkan oIeh \nKakorlantas PoIri; \n12. hasil Cek Fisik Ranmor; dan \n13. tanda bukti pendaftaran BPKB. "
    },
    {
        "page": 44,
        "content": "-45-\nPasa150 \nPenerbitan STNK baru untuk Ranrnor hasil lelang \npenghapusan Ranrnor dinas Tentara Nasional \nIndonesia/Polri harus rnernenuhi persyaratan: \na. rnengisi forrnulir perrnohonan; \nb. rnelarnpirkan: \n1. Kartu Tanda Penduduk; \n2. surat kuasa berrneterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat keputusan penghapusan Ranrnor dan daftar \npenghapusan Ranrnor dari dinas Tentara Nasional \nIndonesia/Polri; \n4. surat penetapan pernenang dan kutipan risalah \nlelang Ranrnor; \n5. berita acara penyerahan Ranrnor yang dilelang; \n6. bukti pernbayaran harga lelang; \n7. hasil Cek Fisik Ranrnor; dan \n8. tanda bukti pendaftaran BPKB. \nPasal51 \nPenerbitan STNK baru untuk hasil le1ang Ranrnor ternuan \nDirektorat Jenderal Bea dan Cukai Kernenterian Keuangan, \nharus rnernenuhi persyaratan: \na. rnengisi forrnulir perrnohonan; \nb. rnelarnpirkan: \n1. Kartu Tanda Penduduk; \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat keputusan penetapan barang milik negara; \n4. kutipan risalah lelang Ranmor yang diterbitkan \noleh balai lelang negara; \n5. berita acara penyerahan Ranmor yang dilelang; \n6. bukti pernbayaran harga le1ang; \n7. SRUT; \n8. hasil Cek Fisik Ranmor; dan "
    },
    {
        "page": 45,
        "content": "-46-\n9. tanda bukti pendaftaran BPKB. \nParagraf2 \nPerubahan Identitas Ranmor dan Pemilik \nPasa152 \nPerubahan data STNK atas dasar perubahan bentuk \nRanmor harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. STNK; \n4. rekomendasi dari unit pelaksana Regident untuk \nperubahan bentuk Ranmor; \n5. surat keterangan dari Agen Pemegang Merek atau \nbengkel umum yang melaksanakan perubahan \nbentuk Ranmor yang disertai TDP/NIB, SIUP, \nNomor Pokok Wajib Pajak dan surat keterangan \ndomisili; \n6. hasil Cek Fisik Ranmor; dan \n7. Tanda bukti pendaftaran BPKB. \nPasa153 \nPerubahan data STNK atas dasar perubahan fungsi \nRanmor, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dima1csud \ndalam Pasa110 ayat (6) huruf a dan huruf b; \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. STNK; "
    },
    {
        "page": 46,
        "content": "-47-\n4. surat izin penyelenggaraan angkutan umum dari \ninstansi yang berwenang, untuk perubahan \nfungsi dari Ranmor perseorangan menjadi \nRanmor angkutan umum; \n5. surat keterangan dari instansi yang berwenang, \nuntuk perubahan fungsi dari Ranmor angkutan \numum menjadi Ranmor perseorangan; \n6. hasil Cek Fisik Ranmor; dan \n7. tanda bukti pendaftaran BPKB. \nPasal54 \nPerubahan data STNK atas dasar perubahan warna \nRanmor, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasall0 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. STNK; \n4. rekomendasi dari unit pelaksana Regident untuk \nperubahan warna Ranmor; \n5. surat keterangan dari bengkel umum yang \nmelaksanakan perubahan warna Ranmor yang \ndisertai TDP/NIB, SIUP, Nomor Pokok Wajib Pajak \ndan surat keterangan domisili; \n6. hasH Cek Fisik Ranmor; dan \n7. Tanda bukti pendaftaran BPKB. \nPasal55 \n(1) Perubahan data STNK atas dasar perubahan mesin \nbaru, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); "
    },
    {
        "page": 47,
        "content": "-48-\n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; \n4. rekomendasi dari unit pelaksana Regident \nuntuk ganti mesin baru; \n5. faktur pembe1ian mesin dari agen pemegang \nmerek; \n6. dokumen pemberitahuan impor barang; \n7. surat keterangan dari bengkel resmi agen \npemegang merek yang melaksanakan \npenggantian mesin yang disertai TDP /NIB, \nSlUP, Nomor Pokok Wajib Pajak dan surat \nketerangan domisili; \n8. hasil Cek Fisik Ranmor; dan \n9. tanda bukti pendaftaran BPKB. \n(2) Perubahan data STNK atas dasar perubahan mesm \nbukan baru dari Ranmor lain, harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; \n4. rekomendasi dari unit pelaksana Regident \nuntuk ganti mesin bukan baru; \n5. surat keterangan dari bengkel resmi agen \npemegang merek atau bengkel umum yang \nmelaksanakan penggantian mesinyang \ndisertai TDP/NIB, SIUP, Nomor Pokok Wajib \nPajak dan surat keterangan domisili; \n6. BPKB dan STNK dari asal usul mesin \npengganti; \n7. hasil Cek Fisik Ranmor; dan "
    },
    {
        "page": 48,
        "content": "-49-\n8. tanda bukti pendaftaran BPKB. \nPasal56 \n(1) Perubahan data STNK atas dasar perubahan NRKB, \nharus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimal<sud \ndalam 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; \n4. hasil Cek Fisik Ranmor; dan \n5. tanda bukti pendaftaran BPKB. \n(2) Selain persyaratan sebagaimana dimaksud pada ayat \n(1), untuk perubahan NRKB rnenjadi NRKB pilihan \nditambah dengan tanda bukti pembayaran \nPenerimaan Negara Bukan Pajak dan Surat \nketerangan NRKB pilihan. \nPasal57 \n(1) Perubahan data STNK atas dasar perubahan nama \ntanpa perubahan pemilik dan alamat, harus \nrnemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melarnpirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalarn Pasal 10 ayat (6); \n2. surat kuasa berrneterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. akta perubahan nama bagi badan hukum; \n4. penetapan pengadilan bagi perorangan; \n5. STNK; \n6. hasil Cek Fisik Ranrnor; dan \n7. tanda bukti pendaftaran BPKB. "
    },
    {
        "page": 49,
        "content": "-50-\n(2) Perubahan data STNK atas dasar perubahan alamat \npemilik Ranmor dalam satu wilayah Regident, harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. al{ta perubahan alamat bagi badan hukum; \n4. STNK; dan \n5. hasil Cek Fisik Ranmor. \n6. tanda bukti pendaftaran BPKB. \n(3) Perubahan data STNK atas dasar perubahan alamat \npemilik ke luar wilayah Regident, memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. akta perubahan alamat bagi badan hukum. \n4. STNK; \n5. tanda bukti pembayaran penerimaan negara \nbukan pajak Mutasi Ranmor keluar wilayah \nRegident; dan \n6. hasil Cek Fisik Ranmor; \n7. tanda bukti pendaftaran BPKB. \nPasal58 \nPerubahan data STNK atas dasar perubahan pemilik \nRanmor harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: "
    },
    {
        "page": 50,
        "content": "-51 -\n1. tanda bukti identitas sebagairnana dirnaksud \ndalam PasallO ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan ; \n3. STNK; \n4. bukti pemindahtanganan kepemilikan; \n5. hasil eek Fisik Ranmor; dan \n6. tanda bukti pendaftaran BPKB. \nPasal59 \nPerubahan data STNK atas dasar perubahan pemilik \nRanmor untuk Ranmor bekas Badan Internasional atau \nPNA, harus memenuhi persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. surat permohonan dari Badan Internasional atau \nPNA; \n4. kuitansi pembelian bermeterai cukup; \n5. surat keterangan pelepasan hak dari Badan \nInternasional atau PNA yang bersangkutan; \n6. STNK; \n7. dokumen kepabeanan, meliputi: \na) formulir B atau Otomasi data B, sebagai \ntanda bukti penangguhan bea masuk atau \nSKB pengganti, untuk pemindahtanganan \nkepemilikan antar Badan Internasional atau \nPNA; atau \nb) formulir C atau Otomasi data C sebagai \ntanda bukti pelunasan bea masuk, untuk \npemindahtanganan kepemilikan kepada \nperorangan/badan hukum; "
    },
    {
        "page": 51,
        "content": "-52 -\n8. rekomendasi dari kementerian luar negeri untuk \nRanmor PNA atau surat rekomendasi Sekretariat \nNegara untuk Ranmor Badan Internasional; \n9. hasil penelitian keabsahan mengenai surat \nketerangan impor Ranmor yang dikeluarkan oleh \nKakoriantas PoIri; \n10. hasil Cek Fisik Ranmor; dan \n11. taI~da bukti pendaftaran BPKB. \nPasal60 \n(1) Dalam hal STNK dan/atau TNKB hilang atau rusak, \npemilik Ranmor dapat mengajukan permohonan \npenggan tian. \n(2) Penggantian STNK karena hilang dilal(sanakan dengan \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwaldikan; \n3. BPKB; \n4. surat pemyataan pemilik bermeterai cukup \nmengenai STNK yang hilang tidak terkait \nkasus pidana, perdata dan/atau pelanggaran \nlal u lin tas; \n5. surat tanda penerimaan laporan dari Polri; \ndan \n6. hasil Cek Fisik Ranmor. \n(3) Penggantian STNK karena rusak harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melrunpirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalrun PasallO ayat (6); "
    },
    {
        "page": 52,
        "content": "2. \n3. \n4. \n5. -53 -\nsurat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \nBPKB; \nSTNK yang rusak; dan \nhasil eek Fisik Ranmor. \n(4) Pengganti~n TNKB karena hilang, harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasall0 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu Tanda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; \n4. Surat Tanda Penerimaan Laporan dari Polri; \ndan \n5. tanda bukti pembayaran penerimaan negara \nbukan pajak. \n(5) Penerbitan TNKB karena rusak, harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasallO ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nkartu tanda penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; \n4. TNKB yang rusak; dan \n5. Tanda bukti pembayaran penerimaan negara \nbukan pajak. "
    },
    {
        "page": 53,
        "content": "-54-\nBagian Ketiga \nPengesahan dan Perpanjangan STNK \nPasal61 \n(1) Pengesahan STNK dapat dilakukan secara: \na. manual pada pelayanan Samsat; atau \nb. elektronik pada pelayanan Samsat Online. \n(2) Pengesahan STNK secara manual sebagaimana \ndimaksud pada ayat (1) huruf a, harus memenuhi \npersyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. tanda bukti identitas sebagaimana dimaksud \ndalam PasaII0 ayat (6); \n2. surat kuasa bermeterai cukup dan fotokopi \nKartu TaI\"lda Penduduk yang diberi kuasa \nbagi yang diwakilkan; \n3. STNK; dan \n4. TBPKP. \n(3) Pengesahan STNK secara elektronik sebagaimana \ndimaksud pada ayat (1) huruf b, harus memenuhi \npersyaratan: \na. Ranmor teregistrasi dalam pangkalan data sistem \ninformasi Regident Ranmor PoIri; \nb. status Ranmor tidak dalam biokir; dan \nc. telah melakukan pembayaran Pajal< Kendaraan \nBermotor (PKB) dan Sumbangan Wajib Dana \nKecelakaan Lalu Lintas Jalan (SWDKLW). \nPenerbitan \npersyaratan: Pasal62 \nSTNK perpanjangan \na. mengisi formulir permohonan; \nb. melampirkan: harus memenuhi \n1. tanda bukti identitas sebagaimana dimaksud \ndalam Pasal 10 ayat (6); "
    },
    {
        "page": 54,
        "content": "-55-\n2. surat kuasa bermeterai cukup dan fotokopi Kartu \nTanda Penduduk yang diberi kuasa bagi yang \ndiwakilkan; \n3. STNK; \n4. BPKB; dan \n5. hasil eek Fisik Ranmor. \nBagian Keempat \nProsedur Penerbitan, Pengesahan, dan Perpanjangan \nSTNK dan/ atau TNKB \nPasal63 \n( 1) Permohonan penerbitan, pengesahan, dan perpanjangan \nSTNK dan/atau TNKB diajukan oleh pemohon STNK \ndan/atau TNKB kepada Unit pelaksana Regident \nPengoperasian Ranmor yang berkedudukan di Kantor \nBersama Samsat. \n(2) Terhadap permohonan sebagaimana dimaksud pada \nayat (1) dilaksanakan melalui tahapan kegiatan: \na. pendaftaran; \nb. penetapan; \nc. penerimaan pembayaran; \nd. pencetakan dan pengesahan; \ne. penyerahan; dan \nf. pengarsipan. \n(3) Tahapan kegiatan sebagaimana dimaksud pada ayat \n(1) dilakukan oleh kelompok kerja. \nPasal64 \n(1) Pendaftaran sebagaimana dimaksud dalam Pasal 63 \nayat (2) huruf a, meruPakan kegiatan penerimaan \npermohonan registrasi. \n(2) Permohonan sebagaimana dimaksud pada ayat (1) \nditerima pada loket pendaftaran oleh kelompok kerja \npendaftaran. \n(3) Permohonan yang diterima sebagaimana dimaksud \npada ayat (2) dilakukan: "
    },
    {
        "page": 55,
        "content": "-56-\na. pendataan dan verifikasi melalui pemeriksaan \nkelengkapan dan keabsahan persyaratan, \npencocokan dan penelitian dokumen persyaratan \ndengan yang tercantum dalam formulir danl atau \nke instansi penerbit dokumen persyaratan; \nb. pemanggilan atau pemasukan data identitas \npemilik dan Ranmor pada sistem informasi \nRegident Ranmor; dan \nc. pencocokan data sebagaimana dimaksud pada \nhuruf b dengan data Regident Kepemilikan \nRanmor secara online. \n(4) Dalam hal permohonan sebagaimana dimaksud pada \nayat (3) tidal{ lengkap dan tidak sah, petugas \nkelompok kerja mengembalikan kepada pemohon \nuntuk dilengkapi. \n(4) Apabila permohonan sudah lengkap dan sah, petugas \npendaftaran menyerahkan kepada petugas kelompok \nkerja penetapan. \nPasal65 \nPenetapan sebagaimana dimal{sud dalam Pasal 63 ayat (2) \nhuruf b, merupakan kegiatan: \na. penetapan penerimaan negara bukan pajak STNK, \nTNKB dan/atau NRKB pilihan yan.g harus dibayarkan; \ndan \nb. penyerahan SKKP kepada petugas kelompok kerja \npenerimaan pembayaran. \nPasal66 \n(1) Penerimaan pembayaran sebagaimana dimaksud \ndalam Pasal 63 ayat (2) huruf c merupal{an kegiatan \npembayaran biaya administrasi STNK dan/atau TNKB \nsesuai Penerimaan Negara Bukan Pajak Polri yang \ndilakukan oleh pemohon registrasi Ranmor melalui \nBank Persepsi secara tunai atau non tunai sesuai \ndengan ketentuan peraturan perundang-undangan. "
    },
    {
        "page": 56,
        "content": "-57-\n(2) Kegiatan pembayaran sebagaimana dimaksud pada \nayat (1) dilakukan setelah petugas kelompok kerja \nmemberitahukan kepada pemohon untuk membayar \nsebagaimana yang tercantum pada SKKP secara \nmanual atau elektronik. \n(3) Terhadap penerimaan pembayaran sebagaimana \ndimaksud pada ayat (1) petugas kelompok kerja \nmelakukan kegiatan: \na. menerima pembayaran dan menyerahkan bukti \nbayar;dan \nb. menyerahkan pembayaran Penerimaan Negara \nBukan Pajal<: kepada Bendahara Penerima atau \nPembantu Bendahara Penerima Polri. \n(4) Dokumen persyaratan sebagaimana dimaksud dalam \nPasal 64 ayat (3) huruf a, diserahkan kepada petugas \nkelompok keIja pencetakan dan pengesahan. \n(5) Penerimaan pembayaran sebagaimana dimaksud pada \nayat (1) disetorkan ke Kas Negara oleh Bendahara \nPenerima atau Pembantu Bendahara Penerima \nPenerimaan Negara Bukan Pajak Polri. \nPasal67 \n(1) Pencetakan dan pengesahan sebagaimana dimal(sud \ndalam Pasal 63 ayat (2) huruf d merupakan kegiatan: \na. pencetakan STNK dan TNKB; \nb. pengesahan STNK dengan cara: \n1. manual dengan pemberian stempel dan/atau \npembubuhan paraf, dan pencantuman \ntanggal bulan tahun pada kolom \npengesahan; atau \n2. elektronik melalui sistem informasi Regident \nRanmor; \nc. pencetakan surat keterangan NRKB pilihan bagi \nyang menggunakan NRKB pilihan pada saat \nperpanjangan STNK. \n(2) STNK danl atau TNKB, dan surat keterangan NRKB \npilihan sebagaimana dimal(sud pada ayat (1) serta "
    },
    {
        "page": 57,
        "content": "-58-\nTBPKP diserahkan kepada petugas kelompok kerja \npenyerahan. \n(3) STNK ditandatangani oleh Direktur Lalu Lintas \nKepolisian Daerah secara elektronik. \nPasal68 \n(1) Penyerahan sebagaimana dimaksud da!am Pasa! 63 \nayat (2) huruf e, merupakan kegiatan menyerahkan \nSTNK dan/ atau TNKB kepada pemohon registrasi. \n(2) Dalam kegiatan penyerahan sebagaimana dimaksud \npada ayat (1), petugas kelompok kerja penyerahan, \nmelakukan kegiatan: \n(3) a. pemisahan STNK, TBPKP, TNKB dan/atau surat \nketerangan NRKB pilihan dari dokumen \npersyaratan registrasi; dan \nb. penyerahan STNK, TBPKP, TNKB dan/atau surat \nketerangan NRKB pilihan kepada pemohon dengan \nmelakukan pencatatan dan pemohon menandatangani \nbuku register penyerahan. \nDokumen \ndimaksud persyaratan registrasi sebagaimana \npada ayat (2) huruf a, diserahkan kepada \npetugas kelompok kerja pengarsipan untuk disimpan \nsecara manual dan/ atau elektronik. \nPasal69 \n(1) Pengarsipan sebagaimana dimaksud dalam Pasal 63 \nayat (2) huruf f merupalmn kegiatan mengarsipkan \ndokumen permohonan penerbitan STNK dan dokumen \npendukung lain secara manual dan/atau elektronik. \n(2) Pengarsipan sebagaimana dimaksud pada ayat (1) \ndilaksanakan dengan mengelompokkan dokumen \nberdasarkan nomor registrasi dan jenis Ranmor. "
    },
    {
        "page": 58,
        "content": "-59 -\nBAB VI \nREGISTRASI RANMOR KHUSUS \nBagian Kesatu \nUmum \nPasal 70 \n(1) Regident Ranmor dapat dilaksanakan secara khusus \nberdasarkan pertimbangan: \na. kepemilikan; \nb. kepentingan; atau \nc. keadaan tertentu. \n(2) Pertimbangan kepemilikan sebagaimana dimaksud \npada ayat (1) huruf a, meliputi Ranmor dinas: \na. Tentara Nasional Indonesia; dan \nb. Polri. \n(3) Pertimbangan kepentingan sebagaimana dimaksud \npada ayat (1) hurufb, meliputi: \na. Ranmor yang digunakan pada kawasan \nperdagangan bebas; \nb. Ranmor yang digunakan pada Kawasan Strategis \nNasional; \nc. Ranmor asing yang digunakan untuk: \n1. angkutan antar negara; \n2. kegiatan pertemuan antar negara, misi \nkemanusiaan, olah raga, pariwisata \ndi Indonesia; \n3. kepentingan sosial ekonomi pada daerah \nyang memiliki keterbatasan infrastruktur \natas rekomendasi dari pemerintah daerah; \ndanjatau \n4. pengamanan pejabat negara asing \nberdasarkan rekomendasi kementerian Iuar \nnegeri danjatau Kementerian Sekretariat \nNegara; \nd. Ranmor yang digunakan untuk Pejabat Konsul \nKehormatan; "
    },
    {
        "page": 59,
        "content": "(4) -60-\ne. Ranmor yang digunakan untuk Pejabat/Petugas \nyang bertugas di bidang intelijen danl atau \npenyidik guna menjaga/menjamin kerahasiaan. \nidentitas, baik diri pribadi maupun sarana yang \ndigunakan; dan \nf. Ranmor dinas Tentara Nasional Indonesia, Polri \ndan instansi pemerintah yang digunakan oleh \npejabat eselon tertentu di lingkungan instansinya \nguna menjamin/memelihara keamanan/pengamanan \nbagi pejabat yang bersangkutan. \nPertimbangan \ndimaksud pada keadaan tertentu sebagaimana \nayat (1) huruf c, meliputi Ranmor \ndalam keadaan kontingensi sebagai akibat bencana \nalam danl atau konflik sosial. \n(5) Ranmor yang telah diregistrasi berdasarkan \npertimbangan kepentingan sebagaimana dimaksud \npada ayat (3) huruf a dan huruf b, diberikan alokasi \nNRKB khusus yang pengoperasiannya di kawasan \nperdagangan bebas dan/atau Kawasan Strategis \nNasional. \n(6) Ranmor yang telah diregistrasi berdasarkan \npertimbangan kepentingan sebagaimana dimaksud \npada ayat (3) huruf c, diberikan bukti Regident \nRanmor berupa: \na. STNK LBN; dan \nb. TNKB LBN. \n(7) Ranmor yang telah diregistrasi berdasarkan \npertimbangan kepentingan sebagaimana dimaksud \npada ayat (3) huruf e, diberikan bukti Regident \nRanmor berupa: \na. STNK Rahasia; dan \nb. TNKB Rahasia. \n(8) Ranmor telah diregistrasi berdasarkan pertimbangan \nkepentingan sebagaimana dimal(sud ayat (3) huruf d \ndan huruf f, diberikan bukti Regident Ranmor berupa: \na. STNK Khusus; dan \nb. TNKB Khusus. "
    },
    {
        "page": 60,
        "content": "-61 ':.-\nBagian Kedua \nSTNK LBN dan TNKB LBN \nParagraf 1 \nUmum \nPasal 71 \n(1) STNK LBN sebagaimana dimaksud dalam Pasal 70 \nayat (6) huruf a, memuat: \na. nomor registrasi; \nb. nama dan alamat pemilik; \nc. merek; \nd. tipe; \ne. jenis; \nf. model; \ng. tahun pembuatan; \nh. isi silinder I daya listrik; \n1. wama; \nj. nomor rangka; \nk. nomor mesin; \nL masa berlaku; \nm. tanggal registrasi; \nn. bahan bakar I sumber energi; \no. kode wilayah; \np. negara asal; \nq. kepentingan pengoperasian; dan \nr. daerah tujuan. \n(2) STNK LEN berlaku selama 30 (tiga puluh) hari dan \ndapat diperpanjang untuk jangka waktu 30 \n(tiga puluh) hari setiap perpanjangan. \n(3) Perpanjangan sebagaimana dimalmud pada ayat (2) \ndilakukan paling banyak 6 (enam) kali perpanjangan. \n(4) STNK LEN sebagaimana dimaksud pada ayat (2) hanya \nberlaku dalam wilayah provinsi dimana STNK LEN \nditerbitkan. "
    },
    {
        "page": 61,
        "content": "-62-\n(5) STNK LBN sebagaimana dimaksud pada ayat (4) \nsesuai dengan standardisasi spesifikasi teknis yang \nditetapkan dengan keputusan Kakorlantas Polri. \n(6) STNK LBN sebagaimana dimaksud pada ayat (5) \nmenggunakan materiel yang diadakan secara terpusat \noleh Korlantas Polri. \nPasal 72 \n(1) TNKB LBN sebagaimana dimaksud dalam Pasal 70 \nayat (6) huruf b, memuat: \na. kode wilayah; \nb. nomor registrasi; \nc. kode pengoperasian; dan \nd. masa beriaku. \n(2) Masa berlaku TNKB LBN sama dengan masa berlaku \nSTNKLBN. \n(3) TNKB LBN berwarna dasar perak dengan tulisan \nhitam. \n(4) TNKB LBN dipasang pada: \na. bagian yang mudah terlihat dan teridentifikasi, \nuntuk sepeda motor; dan \nb. bagian kaca depan dan belakang sebelah kiri atas, \nuntuk Ranmor selain sepeda motor. \n(5) TNKB LBN sebagaimana dimaksud pada ayat (4) \nsesuai dengan standardisasi spesifikasi teknis yang \nditetapkan dengan keputusan Kakorlantas Polri. \n(6) TNKB LBN sebagaimana dimaksud pada ayat (5) \nmenggunakan materiel yang diadakan secara terpusat \noleh Korlantas Polri. \nParagraf2 \nPersyaratan Penerbitan \nPasal 73 \n(1) Setiap Ranmor Asing yang almn dioperasionalkan \ndi wilayah negara Republik Indonesia dilaksanakan "
    },
    {
        "page": 62,
        "content": "-63-\nRegident Ranmor dengan menerbitkan STNK LBN dan \nTNIill LBN. \n(2) Penerbitan STNK LBN dan TNKB LBN sebagaimana \ndimaksud pada ayat (1) dilaksanakan oleh Unit \npelaksana Regident Ranmor. \n(3) Untuk memperoleh STNK LBN dan TNKB LBN \nsebagaimana dimaksud pada ayat (1), harus \nmemenuhi persyaratan: \na. mengisi formulir permohonan STNK LBN yang \nberisi: \n1. identitas pemilik dan Ranmor Asing; dan \n2. kepentingan penggunaan Ranmor di wilayah \nIndonesia, dan wilayah penggunaannya; \nb. melampirkan dokumen tanda bukti kepemilikan \ndan/ atau pemberi legitimasi pengoperasian \nRanmor yang diterbitkan oleh pejabat yang \nberwenang di negara asal Ranmor; dan \nc. surat rekomendasi dari kementerian/lembaga/ \npemerintah daerah tentang kepentingan \npenggunaan Ranmor asing. \nParagraf3 \nProsedur Penerbitan dan Perpanjangan \nPasa174 \n(1) Permohonan penerbitan dan perpanjangan STNK LBN \ndan TNKB LBN diajukan kepada Unit Pelaksana \nRegident Ranmor. \n(2) Unit pelaksana Regident Ranmor sebagaimana \ndimaksud pada ayat (1) melaksanakan: \na. penelitian dan verifikasi dokumen persyaratan; \nb. pemasukan data identitas pemilik dan Ranmor \npada buku register secara manual dan/ atau \nelektronik pada sistem informasi Regident \nRanmor; \nc. pemberitahuan kepada pemohon untuk \nmelakukan pembayaran penerimaan negara "
    },
    {
        "page": 63,
        "content": "-64-\nbukan pajak STNK LBN dan TNKB LBN serta \nSumbangan Wajib Dana Kecelakaan Lalu Lintas \nJalan (SWDKLW) ke Bank Persepsi atau \nBendahara Penerimaan; \nd. pencetakan dan penerbitan STNK LBN dan TNKB \nLBN; \ne. penyerahan STNK LBN kepada pemohon serta \npemasangan TNKB LBN; dan \nf. pengarsipan dokumen STNK LBN. \n(3) Pelaksanaan pelayanan STNK LBN dan TNKB LBN \nditetapkan dengan keputusan Kakorlan.tas Polri. \nBagian Ketiga \nSTNK dan TNKB Khusus atau Rahasia \nParagraf 1 \nUmum \nPasal 75 \n(1) STNK Rahasia dan STNK Khusus sebagaimana \ndimaksud dalam Pasal 70 ayat (7) huruf a dan ayat (8) \nhuruf a, paling sedikit memuat: \na. NRKB; \nb. nama pemilik; \nc. NIK/TDP/NIB/ kartu izin tinggal tetap/kartu izin \ntinggal semen tara; \nd. alamat pemilik; \ne. merek; \nf. tipe; \ng. jenis; \nh. model; \n1. tahun pembuatan; \nJ. isi sHinder / daya listrik; \nk. warna; \nl. nomor rangka; \nm. nomor mesin; \nn. nomor BPKB; "
    },
    {
        "page": 64,
        "content": "-65-\no. rnasa berlaku; \np. warna TNKB; \nq. tahun registrasi; \nr. bahan bakar/sumber energi; \ns. kode lokasi; dan \nt. nomor urut register. \n(2) STNK Khusus dan STNK Rahasia berlaku selama \n1 (satu) tahun sejak tanggal diterbitkan. \nParagraf2 \nPersyaratan Penerbitan \nPasal76 \n(1) Persyaratan penerbitan STNK/TNKB Khusus: \na. surat rekomendasi dari: \n1. Direktur Intelijen Keamanan Kepolisian \nDaerah Metro Jaya, untuk tingkat pusat; \n2. Direktur Intelijen Keamanan Kepolisian \nDaerah, untuk tingkat provinsi/kabupaten/ \nkota; \n3. Kepala Divisi Profesi dan Pengamanan Polri, \nuntuk tingkat markas besar Polri atau \nKepala Bidang Profesi dan Pengamanan \nKepolisian Daerah untuk tingkat Kepolisian \nDaerah; atau \n4. Kementerian Luar Negeri untuk Pejabat \nKonsul Kehormatan; \nb. fotokopi STNK Ranmor dinas; \nc. STNK atas nama pribadi untuk Pejabat Konsul \nKehormatan; \nd. fotokopi BPKB untuk Ranmor dinas milik Instansi \nPemerintah, kecuali Ranmor dinas Tentara \nNasional Indonesia/Polri; \ne. BPKB atas nama pribadi untuk Pejabat Konsul \nKehormatan; "
    },
    {
        "page": 65,
        "content": "-66-\nf. fotokopi kartu tanda penduduk dan Kartu Tanda \nAnggota atau Kartu Pegawai pejabat pengguna \nRanmor dinas; \ng. kartu tanda penduduk dan kartu identitas Konsul \nKehormatan untuk Pejabat Konsul Kehormatan; \nh. fotokopi keputusan jabatan Pejabat Pengguna \nRanmor dinas; \ni. STNK khusus yang lama, bagi Ranmor dinas yang \nperuah diberikan STNK/TNKB khusus; dan \nj. hasil Cek Fisik Ranmor. \n(2) Persyaratan penerbitan STNK/TNKB rahasia: \na. surat rekomendasi dari: \n1. Direktur Intelijen Keamanan Kepolisian \nDaerah Metro Jaya, untuk tingkat pusat; \n2. Direktur Intelijen Keamanan Kepolisian \nDaerah, untuk tingkat provinsi/kabupaten/ \nkota; atau \n3. Kepala Divisi Profesi dan Pengamanan Polri, \nuntuk tingkat markas besar Polri atau \nKepala Bidang Profesi dan Pengamanan \nKepolisian Daerah, untuk tingkat Kepolisian \nDaerah; \nb. fotokopi STNK atau STNK Dinas Ranmor \nTNI/Polri; \nc. fotokopi BPKB untuk Ranmor dinas milik instansi \npemerintah, kecuali Ranmor dinas TNI/Polri; \nd. fotokopi Kartu Tanda Penduduk dan Kartu Tanda \nAnggota atau Kartu Pegawai Pejabat Pengguna \nRanmor dinas; \ne. fotokopi keputusan jabatan Pejabat Pengguna \nRanmor dinas; \nf. Surat tugas dari instansi yang bersangkutan; \ng. STNK Rahasia yang lama, bagi Ranmor dinas \nyang peruah diberikan STNK/TNKB Rahasia; dan \nh. hasil Cek Fisik Ranmor. "
    },
    {
        "page": 66,
        "content": "-67-\nParagraf 3 \nProsedur Penerbitan \nPasal77 \n(1) Permohonan STNK dan TNKB Rahasia atau Khusus \ndisampaikan kepada Petugas kelompok kerja \npendaftaran di Direktorat Lalu Lintas Kepolisian \nDaerah. \n(2) Kelompok keIja pendaftaran sebagaimana dimaksud \npada ayat (1), melakukan: \na. pendataan dan verifikasi melalui pemeriksaan \nkelengkapan dan keabsahan persyaratan; dan \nb. pencocokan dan penelitian dokumen persyaratan \ndengan yang tercantum dalam formulir dan/atau \ninstansi penerbit dokumen persyaratan. \n(3) Apabila persyaratan yang diajukan tidak lengkap, \npetugas kelompok kerja pendaftaran memberitahukan \nsecara lisan/tertulis dan mengembalikan kepada \npemohon untuk melengkapi kekurangan persyaratan. \n(4) Dalam hal dokumen persyaratan sudah lengkap dan \nsah, petugas kelompok kerja pendaftaran: \na. melakukan pengecekan data dengan data \nRegident Kepemilikan Ranmor secara online; \nb. melakukan penetapan NRKB; \nc. memasukan data fisik berupa identitas dan fungsi \nRanmor serta identitas pemilik secara manual \ndan elektronik; dan \nd. memberitahukankepada pemohon untuk \nmelakukan pembayaran Penerimaan Negara \nBukan Pajak. \n(5) Petugas pada kelompok keIja pencetakan melakukan: \na. penerbitan STNK dan pencetakan TNKB Rahasia \natau Khusus; dan \nb. penyerahan STNK dan TNKB Rahasia atau \nKhusus kepada kelompok kerja penghimpunan \ndan penggabungan serta penyerahan. "
    },
    {
        "page": 67,
        "content": "-68-\n(6) Kelompok kerja penyerahan melakukan: \na. penyerahan STNK dan TNKB Rahasia atau Khusus \nkepada pemohon dengan melakukan pencatatan \ndan pemohon menandatangani buku register \npenyerahan; dan \nb. penyerahan dokumen arsip kepada kelompok \nkerja pengarsipan. \n(7) Petugas kelompok kerja pengarsipan, melakukan: \na. penyimpanan arsip; dan \nb. pencatatan dan pendataan arsip secara manual \natau elektronik. \n(8) STNK dan TNKB Rahasia atau Khusus berlaku selama \n1 (satu) tahun dan dapat diperpanjang dengan \nmemenuhi persyaratan sebagaimana dimaksud dalam \nPasa176. \n(9) Penerbitan STNK dan TNKB Rahasia atau Khusus \ndipungut biaya penerimaan negara bukan pajak sesuai \ndengan ketentuan peraturan perundang-undangan. \nBAB VII \nSTCK DAN TCKB \nBagian Kesatu \nUmum \nPasal 78 \n(1) Setiap Ranmor yang belum diregistrasi dapat \ndioperasikan di jalan untuk kepentingan tertentu \ndengan dilengkapi STCK dan TCKB. \n(2) Kepentingan tertentu sebagaimana dimaksud pada \nayat (1) meliputi: \na. memindahkan Ranmor baru dari tempat penjual, \ndistributor atau pabrikan ke tempat tertentu \nuntuk mengganti atau melengkapi komponen \npenting dari kendaraan yang bersangkutan atau \nke tempat pendaftaran Ranmor; "
    },
    {
        "page": 68,
        "content": "-69-\nb. memindahkan dari suatu temp at penyimpanan \ndi suatu pabrik ke tempat penyimpanan di pabrik \nlain; \nc. mencoba Ranmor baru sebelum dijual; \nd. mencoba Ranmor baru yang sedang dalam \npenelitian; atau \ne. memindahkan Ranmor dari tempat penjual \nke tempat pembeli. \nPasal 79 \n(1) STCK dan TCKB diberikan kepada badan usaha \ndi bidang penjualan, pembuatan, perakitan, atau \nimpor Ranmor serta lembaga penelitian di bidang \nRanmor. \n(2) Ranmor yang dilengkapi STCK dan TCKB \ndioperasionalkan oleh petugas badan usaha dengan \nmenggunakan seragam paling banyak 3 (tiga) orang. \n(3) STCK sebagaimana dimaksud pada ayat (1), memuat \ndata: \na. nomoI' registrasi; \nb. nama penanggungjawab; \nc. nama badan usaha; \nd. alamat badan usaha; \ne. kode lokasi; dan \nf. nomoI' urut pendaftaran. \n(4) Setiap STCK harus dilengkapi formulir STCK yang \nberisi data: \na. NomoI' Registrasi; \nb. maksud dan tujuan penggunaan STCK dan \nTCKB; \nc. asal; \nd. tujuan; \ne. meI'ek; \nf. tipe; \ng. jenis; \nh. model; \n1. tahun pembuatan; "
    },
    {
        "page": 69,
        "content": "-70-\nj. isi silinder / daya listrik; \nk. nomor rangka; \n1. nomor mesin; \nm. bahan bakar; \nn. warna; \no. Nomor SUT dan/atau SRUT; \np. masa berlaku; dan \nq. tanda tangan pejabat badan usaha. \n(5) STCK dan formulir STCK merupakan kesatuan yang \ntidak terpisahkan. \n(6) STCK tidak berlaku apabila tidak dilengkapi Formulir \nSTCK. \n(7) STCK sebagaimana dimaksud pada ayat (3) diisi oleh \npetugas Polri secara elektronik. \n(8) Formulir STCK sebagaiman.a dimaksud pada ayat (4) \ndiisi oleh agen pemegang merek/importir umum \nsecara manual atau elektronik. \n(9) STCK dan formulir STCK sebagaimana dimaksud pada \nayat (4) sesuai dengan standardisasi spesifikasi teknis \nyang ditetapkan dengan keputusan Kakorlantas Polri. \n(10) STCK dan formulir STCK sebagaimaIla dimaksud pada \nayat (9), menggunakan materiel yang diadakan secara \nterpusat oleh Korlantas Polri. \n(11) STCK sebagaimana dimaksud pada ayat (4) berlaku \n14 (empat belas) hari sejak diterbitkan dan dapat \ndiperpanjang sebelum habis masa berlaku. \nPasa180 \n(1) TCKB sebagaimana dimaksud dalam Pasal 78 ayat (1), \nmemuat data: \na. kode wilayah; \nb. nomor/angka; dan \nc. seri huruf. \n(2) TCKB dipasang pada tempat yang disediakan di bagian \ndepan dan belakang Ranmor yang mudah terlihat dan \nteridentifikasi. "
    },
    {
        "page": 70,
        "content": "-71 -\n(3) TCKB berlaku selarna badan usaha masih melakukan \nusaha sebagaimana dimaksud dalam Pasal 79 ayat (1). \n(4) Penerbitan TCKB dipungut biaya penerimaan negara \nbukan pajak sesuai dengan ketentuan peraturan \nperundang-undangan. \n(5) TCKB berwarna dasar putih dan tulisan merah. \n(6) TCKB sebagaimana dimaksud pada ayat (1), sesuai \ndengan standardisasi spesifikasi teknis yang \nditetapkan dengan keputusan Kakorlantas Polri. \n(7) TCKB sebagaimana dimaksud pada ayat (5), \nmenggunakan materiel yang diadakan secara terpusat \noleh Korlantas Polri. \nBagian Kedua \nPersyaratan \nPasal81 \nSTCK dan TCKB sebagaimana dimaksud dalam Pasal 78 \nayat (1), diterbitkan dengan persyaratan: \na. mengisi formulir permohonan; \nb. melampirkan: \n1. surat kuasa bermeterai cukup dari badan usaha; \n2. Kartu Tanda Penduduk yang diberi kuasa; \n3. surat keterangan domisili perusahaan; \n4. TDP/NIB; \n5. Nomor Pokok Wajib Pajak; dan \n6. STCK dan formulir STCK lama, bagi yang pernah \nmengajukan permohonan. \nBagian Ketiga \nProsedur Penerbitan \nPasa182 \n(1) Prosedur penerbitan STCK dan TCKB dilaksanakan \nmelalui kelompok kelja pada Unit Pelaksana Regident \nPengoperasian Ranmor meliputi: \na. pendaftaran; "
    },
    {
        "page": 71,
        "content": "-72 -\nb. penerimaan pembayaran; \nc. pencetakan; \nd. penghimpunan, penggabungan dan penyerahan; \ndan \ne. pengarsipan. \n(2) Prosedur penerbitan STCK dan TCKB: \na. pengajuan permohonan dengan melampirkan \ndokumen persyaratan sebagaimana dimaksud \ndalam Pasal 81, kepada kelompok keIja \npendaftaran; \nb. kelompok keIja pendaftaran, melakukan kegiatan: \n1. meneliti kelengkapan dan keabsahan \ndokumen persyaratan dan kelaikan Ranmor; \n2. me1akukan pengecekan dokumen persyaratan; \n3. memasukan data identitas pemohon secara \nmanual danj atau elektronik dalam sistem \ninformasi Regident Ranmor; \n4. menentukan nomor TCKB; \n5. menyampaikan dokumen persyaratan \nkepada kelompok kerja pencetakan; dan \n6. memberitahukan kepada kelompok kerja \npenerimaan pembayaran penerimaan negara \nbukan pajak; \nc. atas dasar pemberitahuan sebagaimana \ndimaksud pada huruf b angka 6, petugas \nkelompok kerja penerimaan pembayaran \nmemberitahukan kepada pemohon atau yang \ndiberi kuasa untuk melakukan pembayaran \npenerimaan negara bukan pajak penerbitan STCK \ndanj atau TCKB melalui Bank Persepsi atau \nBendahara Penerimaan. \nd. setelah menerima dokumen persyaratan beserta \ntanda bukti pembayaran penerimaan negara \nbukan pajak, petugas kelompok kerja \npencetakan, melakukan: "
    },
    {
        "page": 72,
        "content": "-73-\nL . pencetakan STCK dan TCKB; dan \n2. penyerahan STCK dan TCKB yang telah \ndiverifikasi kepada kelompok kerja \npenghimpunan, penggabungan dan penyerahan; \ne. kelompok kerja penghimpunan, penggabungan \ndan penyerahan melakukan: \n1. pemisahan STCK danl atau TCKB dengan \ndokumen persyaratan; \n2. penyerahan STCK danl atau TCKB kepada \npemohon; dan. \n3. penyerahan dokumen persyaratan kepada \nke1ompok kerja pengarsipan; \nf. ke1ompok kerja pengarsipan me1akukan: \n1. pencatatan dan pendataan arsip; dan \n2. penataan dan penyimpanan arsip secara \nmanual dan/atau elektronik. \n(3) STCK sebagaimana dimaksud pada ayat (2) huruf d \nangka 1 ditandatangani secara elektronik oleh \nDirektur Lalu Lintas Kepolisian Daerah. \nPasa183 \nDalam hal STCK dan/atau TCKB hilang atau rusak dapat \ndiajukan permohonan penggantian dengan persyaratan: \na. surat tanda penerimaan 1aporan; \nb. surat kuasa bermeterai cukup dari badan usaha; \nc. Kartu Tanda Penduduk yang diberi kuasa; \nd. surat keterangan domisili perusahaan; \ne. TDP/NIB; dan \nf. Nomor Pokok Wajib Pajak. "
    },
    {
        "page": 73,
        "content": "-74-\nBAB VIII \nPENGHAPUSAN DAN PEMBLOKIRAN REGIDENT RANMOR \nBagian Kesatu \nPenghapusan \nPasaI 84 \n(1) Ranmor yang telah diregistrasi sebagaimana dimaksud \ndaIam Pasal 5 ayat (1) dapat dihapus dari daftar \nRegident Ranmor atas dasar: \na. permintaan pemilik Ranmor; atau \nb. pertimbangan pejabat Regident Ranmor. \n(2) Penghapusan dari daftar Regident Ranmor atas dasar \npermintaan pemilik sebagaimana dimaksud pada ayat \n(1) huruf a, dilakukan terhadap Ranmor yang tidak \ndioperasikan lagi. \n(3) Penghapusan dari daftar Regident Ranmor atas dasar \npertimbangan pejabat di bidang Regident Ranmor \nsebagaimana dimaksud pada ayat (1) huruf b \ndilakukan jika: \na. Ranmor rusak berat sehingga tidak dapat \ndioperasikan; atau \nb. pemilik Ranmor tidak melakukan registrasi ulang \nsekurang-kurangnya 2 (dua) tahun setelah habis \nmasa berlaku STNK. \n(4) Penghapusan dari daftar Regident Ranmor terhadap \nRanmor rusak berat sebagaimana dimaksud pada ayat \n(3) huruf a, tidak berlaku, apabila Ranmor masih \ndaIarn perbaikan berdasarkan surat keterangan dari \nbengkel. \n(5) Penghapusan dari daftar Regident Ranmor terhadap \nRanmor sebagaimana dimaksud pada ayat (3) huruf b, \ntidak berlaku, apabila Ranmor: \na. diblokir; \nb. daIarn proses lelang; atau \nc. Ranmor yang rusak berat masih dalam perbaikan \nberdasarkan surat keterangan dari bengkel. "
    },
    {
        "page": 74,
        "content": "-75-\nPasal 85 \n(1) Sebelum penghapusan dari daftar Regident Ranmor \nberdasarkan pertirnbangan sebagairnana dirnaksud \ndalam Pasal 84 ayat (3), Unit Pelaksana Regident \nRanmor menyampaikan: \na. peringatan pertama, 3 (tiga) bulan sebelum \nmelakukan penghapusan data Regident Ranmor; \nb. peringatan kedua untuk jangka waktu 1 (satu) \nbulan sejak peringatan pertama, apabila pemilik \nRanmor tidak memberikan jawaban/tanggapan; \ndan \nc. peringatan ketiga untuk jangka waktu 1 (satu) \nbulan sejak peringatan kedua, apabila pemilik \nRanmor tidak memberikanjawaban/tanggapan. \n(2) Dalam hal pemilik Ranmor tidak memberikan \njawaban/tanggapan dalam jangka waktu 1 (satu) \nbulan sejak peringatan ketiga, dilakukan penghapusan \nRegident Ranmor. \n(3) Peringatan sebagaimana dimalcsud pada ayat (1) \ndisampaikan secara manual atau elektronik. \nPasa186 \n(1) Permintaan penghapusan Regident Ranmor oleh \npemilik, diajukan dengan melampirkan: \na. surat permohonan; \nb. bukti identitas pemilik Ranmor; \nc. surat pernyataan bermeterai cukup dari pemilik \nRanmor yang menyatakan alasal1. Ranmor tidalc \ndioperasikan; \nd. BPKB; \ne. STNK; \nf. TNKB; dan \ng. foto Ranmor. \n(2) Penghapusan Regident Ranmor dilalmkan dengan \nmemberikan catatan atau tanda cap stempel \n\"DIHAPUS\" pada kartu induk, buku register, BPKB, \nSTNK dan pada sistem manajemen Registrasi Ranmor. "
    },
    {
        "page": 75,
        "content": "-76-\n(3) Registrasi Ranrnor yang sudah dinyatakan dihapus \nsebagaimana dimaksud pada ayat (1) tidak dapat \ndiregistrasi kembali. \nBagian Kedua \nPemblokiran \nPasa187 \n(1) Unit Pelaksana Regident Ranmor dapat melakukan \npemblokiran data BPKB danfatau data STNK. \n(2) Pemblokiran data BPKB sebagaimana dimaksud pada \nayat (1) dilakukan untuk kepentingan: \na. pencegahan perubahan identitas Ranmor dan \npemilik; \nb. penegakan hukum; dan \nc. perlindungan kepentingan kreditur. \n(3) Pemblokiran data STNK sebagaimana dimaksud pada \nayat (1) dilakukan untuk kepentingan: \na. pencegahan pengesahan dan perpanjangan \nRegident Ranmor danfatau penggantian STNK; \ndan \nb. penegakan hukum pelanggaran lalu lintas. \n(4) Permintaan pemblokiran data BPKB danfatau data \nSTNK untuk kepentingan sebagaimana dimaksud pada \nayat (2) dan ayat (3), diajukan oleh: \na. penyidik atau penuntut umum; \nb. panitera berdasarkan penetapan hakim atau \nputusan pengadilan; \nc. kreditur dengan me1ampirkan fotokopi Sertifikat \nFidusia; atau \nd. pemilik Ranmor dengan \npermohonan bermeterai melampirkan \ncukup dan \npemindahtanganan kepemilikan. surat \nbukti \n(5) Permintaan pemblokiran data STNK untuk \nkepentingan sebagaimana dimaksud pada ayat (3), \ndiajukan oleh penyidik laIu lintas terhadap: "
    },
    {
        "page": 76,
        "content": "-77-\na. Ranmor yang diduga terlibat kecelakaan lalu \nlintas dan melarikan diri; atau \nb. Ranmor yang terlibat pelanggaran lalu lintas. \n(6) Permintaan Pemblokiran data STNK untuk \nkepentingan sebagaimana dimaksud pada ayat (3) \nhuruf a, diajukan oleh pemilik Ranmor karena \nperubahan pemilik dengan melampirkan bukti \npemindahtanganan kepemilikan. \nPasal88 \n(1) Pemblokiran data BPKB dan/atau data STNK \nsebagaimana dimaksud dalam Pasal 87 dilaksanakan \ndengan: \na. melakukan verifikasi terhadap data Regident \nRanmor dengan surat permohonan pemblokiran \nberdasarkan permintaan sebagaimana dimaksud \ndalam Pasal 87 ayat (4); \nb. berdasarkan persetujuan pejabat yang berwenang \nme1akukan pemblokiran pada sistem informasi \nRegident Ranmor dan buku register dengan \nmemberikan catatan \"DIBLOKIR\" serta \nmencantumkan alasan, nomoI' dan tanggal surat \npemohon; \nc. mengeluarkan surat keterangan blokir yang \nditandatangani oleh pejabat yang berwenang; dan \nd. melakukan pengarsipan dokumen blokir data \nRegident Ranmor secara manual dan/atau \nelektronik. \n(2) Pejabat yang berwenang sebagaimana dimaksud pada \nayat (1) huruf b dan huruf c, meliputi: \na. Direktur Registrasi dan Identifikasi, untuk \npelayanan Regident Ranmor tingkat markas besar \nPolri; \nb. Direktur Lalu Lintas Kepolisian Daerah, untuk \npelayanan Regident Ranmor tingkat Kepolisian \nDaerah; atau "
    },
    {
        "page": 77,
        "content": "-78-\nc. Kepala Satuan Lalu Lintas, untuk pelayanan \nRegident Ranmor tingkat Kepolisian Resor. \nPasal89 \n(1) Pemblokiran data BPKB dan/atau data STNK dapat \ndibuka berdasarkan permintaan pihal( yang \nmengajukan blokir. \n(2) Dalam hal pemblokiran data STNK atas permintaan \npemilik Ranmor karena pemindahtanganan \nkepemilikan sebagaimana dimaksud dalam Pasal 87 \nayat (6), dapat dibuka dengan proses Regident \nperubahan pemilik Ranmor ke pemilik Ranmor yang \nbaru. \n(3) Buka blokir sebagaimana dimaksud pada ayat (1), \ndilaksanakan oleh Unit Pelaksana Regident Ranmor. \nPasal90 \n(1) Buka blokir data BPKB dan/atau data STNK \nsebagaimana dimaksud dalam Pasal 89 ayat (1) \ndilal(sanakan dengan.: \na. melakukan verifikasi terhadap data Regident \nRanmor dengan surat permohonan buka blokir; \nb. berdasarkan persetujuan pejabat yang berwenang \nmelakukan buka blokir pada sistem informasi \nRegident Ranmor dan buku register dengan \nmemberikan catatan \"BUKA BLOKIR\" serta \nmencantumkan alasan, nomor dan tanggal surat \npemohon; \nc. mengeluarkan surat keterangan buka blokir yang \nditandatangani oleh pejabat yang berwenang; dan \nd. melakukan pengarsipan dokumen buka blokir \ndata Regident Ranmor secara manual danl atau \nelektronik \n(2) Pejabat yang berwenang sebagaimana dimal<sud pada \nayat (1) hurufb dan hurufc, meliputi: "
    },
    {
        "page": 78,
        "content": "-79-\na. Direktur Registrasi dan Identifikasi, untuk \npelayanan Regident Ranmor tingkat markas besar \nPolri; \nb. Direktur Lalu Lintas Kepolisian Daerah, untuk \npelayanan Regident Ranmor tingkat Kepolisian \nDaerah; atau \nc. Kepala Satuan Lalu Lintas, untuk pelayanan \nRegident Ranmor tingkat Kepolisian Resor. \nBABIX \nKETENTUAN PENUTUP \nPasal91 \nPada saat Peraturan Kepolisian ini mulai berlaku: \na. Peraturan Kepala Kepolisian Negara Republik \nIndonesia Nomor 4 Tahun 2006 tentang Penomoran \nKendaraan Bermotor; dan \nb. Peraturan Kepala Kepolisian Negara Republik \nIndonesia Nomor 5 Tahun 2012 tentang Registrasi dan \nIdentifikasi Kendaraan Bermotor (Berita Negara \nRepublik Indonesia Tahun 2012 Nomor 209), \ndicabut dan dinyatakan tidak berlaku. \nPasal92 \nPeraturan Kepolisian ini mulai berlaku pada tanggal \ndiundangkan. "
    },
    {
        "page": 80,
        "content": "-81 -\nLAMPIRAN \nPERATURAN KEPOLISIAN NEGARA REPUBLIK \nINDONESIA \nNOMOR 7 TAHUN 2021 \nTENTANG \nREGISTRASI DAN IDENTIFlKASI KENDARAAN \nBERMOTOR \nA. Penentuan kode wilayah berdasarkan wilayah Regident Ranmor adaIah \nsebagai berikut: \nNO DAERAH KODE LINGKUP WILAYAH PENOMORAN WILAYAH \nI. Provinsi Aceh BL I. Kota Banda Aceh \n2. Kota SubulussaIam \n3. Kota Langsa \n4. Kota Lhokseumawe \n5. Kota Sabang \n6. Kab. Aceh Barat \n7. Kab. Aceh Barat Daya \n8. Kab. Aceh Besar \n9. Kab. Aceh Jaya \n10. Kab. Aceh Selatan \nII. Kab. Aceh Singkil \n12. Kab. Aceh Tamiang \n13. Kab. Aceh Tengah \n14. Kab. Aceh Tenggara \n15. Kab. Aceh Timur \n16. Kab. Aceh Utara \n17. Kab. Bener Meriah \n18. Kab. Bireun \n19. Kab. Gayo Lues \n20. Kab. Nagan Raya \n2I. Kab. Pidie \n22. Kab. Pidie Jaya \n23. Kab. Simeulue "
    },
    {
        "page": 81,
        "content": "-82-\n2. Provinsi BK 1. Kodya Medan \nSumatera 2. Kab. Deli Serdang \nUtara 3. Kota Tebing Tinggi \n4. Kab. Langkat \n5. Kota Binjai \n6. Kab. Simalungun \n7. Kota Pematang Siantar \n8. Kab. Tanah Karo \n9. Kab.Asahan \n10. Kab. Labuhan Batu \nII. Kab. Serdang Begadai \n12. Kab. Batubara \n13. Kota Tanjung Balai \n14. Kab. Labuhan Batu Utara \n15. Kab. Labuhan Batu Selatan \nBB l. Kab.Tapanuli Utara \n2. Kab.Tapanuli Tengah \n3. Kota Sibolga \n4. Kab. Tapanuli Selatan \n5. Kab. Dairi \n6. Kab. Nias \n7. Kab. Humbang Hasundutan \n8. Kab. Samosir \n9. Kab. Toba Samosir \n10. Kota Padang Sidempuan \n11. Kab. Paluta \n12. Kab. Pal as \n13. Kab. Mandailing Natal \n14. Kota Gunung Sitoli \n15. Kab. Nias Barat \n16. Kab. Nias Utara \n17. Kab. Nias Selatan \n18. Kab. Sidikalang \n19. Kab. Pakpak Barat "
    },
    {
        "page": 82,
        "content": "-83-\n3. Provinsi BA l. Kota Padang \nSumatera 2. Kota Bukit Tinggi \nBarat 3. Kota Pandang Panjang \n4. Kota Pariaman \n5. Kota Payakumbuh \n6. Kota Sawahlunto \n7. Kota Solok \n8. Kab. Agam \n9. Kab. Dharmasraya \n10. Kab. Limapuluhkota \n11. Kab. Kep. Mentawai \n12. Kab. Padang Pariaman \n13. Kab. Pasaman \n14. Kab. Pasaman Barat \n15. Kab. Pesisir Selatan \n16. Kab. Sawah lunto Sijunjung \n17. Kab. Solok \n18. Kab. Solok Selatan \n19. Kab. Tanah Datal' \n4. Provinsi Riau BM 1. Kota Pekanbaru \n2. Kab. Indragiri Hulu \n3. Kab. Indragiri HiliI' \n4. Kab. Kampar \n5. Kab. Bengkalis \n6. KotaDumai \n7. Kab. Siak \n8. Kab. Rokan Hulu \n9. Kab. Rokan HiliI' \n10. Kab. Pelalawan \n11. Kab. Kuantan Singingi \n12. Kab. Kep Meranti \n5. Provinsi BP 1. Kab. Karimun \nKepulauan 2. Kab. Bintal1. (Kep Riau) \nRiau 3. Kab. Natuna \n4. Kab. Lingga \n5. KotaBatam "
    },
    {
        "page": 83,
        "content": "-84-\n6. Kota Tanjungpinang \n7. Kab. Kep. Anambas \n6. Provinsi BG 1. Kota Palembang \nSumatera 2. Kota Lubuk Linggau \nSelatan 3. Kota Pagar Alam \n4. Kota Prabumulih \n5. Kab. Banyuasin \n6. Kab. Lahat \n7. Kab. Empat Lawang \n8. Kab. Muara Enim \n9. Kab. Musi Banyuasin \n10. Kab. Musi Rawas \n11. Kab. Musi Rawas Utara \n12. Kab. Ogan Ilir \n13. Kab. Ogan Komering Ilir \n14. Kab. Ogan Komering Ulu \n15. Kab. OKU Selatan \n16. Kab. OKU Timur \n17. Kab. Penukal Abab Lematang \nIlir (PALl) \n7. Provinsi BN 1. Kota Pangkalpinang \nKepuIauan 2. Kab.Bangka \nBangka- 3. Kab. Belitung \nBelitung 4. Kab. Bangka Barat \n5. Kab. Bangka Selatan \n6. Kab. Bangka Tengah \n7. Kab. Belitung Timur \n8. Provinsi BE 1. Kota Bandar Lampung \nLampung 2. Kota Metro \n3. Kab. Lampung Selatan \n4. Kab. Lampung Tengah \n5. Kab. Lampung Utara \n6. Kab. Lampung Barat \n7. Kab. Lampung Timur \n8. Kab. Tanggamus \n9. Kab. Tulang Bawang "
    },
    {
        "page": 84,
        "content": "-85-\n10. Kab. Way Kanan \nII. Kab. Pesawaran \n12. Kab. Pringsewu \n13. Kab. Mesuji \n9. Provinsi BD I. Kota Bengkulu \nBengku1u 2. Kab. Bengkulu Utara \n3. Kab. Bengku1u Se1atan \n4. Kab. Rejang Lebong \n5. Kab.Kaur \n6. Kab. Kepahiang \n7. Kab. Muko-Muko \n8. Kab.Lebong \n9. Kab. Se1uma \n10. Kab. Bengku1u Tengah \n10. Provinsi BH I. KotaJambi \nJambi 2. Kota Sungai Penuh \n3. Kab. Batanghari \n4. Kab. Bungo \n5. Kab. Tebo \n6. Kab. Kerinci \n7. Kab. Tanjung Jabung Barat \n8. Kab. Tanjung Jabung Timur \n9. Kab. Saro1angun \n10. Kab. Merangin \nII. Kab. Muaro Jambi \nII. Provinsi DKI B 1. Daerah Khusus Ibukota Jakarta \nJakarta 2. Kab. Kepu1auan Seribu \n3. Kota Depok \n4. Kota Bekasi \n5. Kab. Bekasi \n6. Kota Tangerang \n7. Kota Tangerang Se1atan \n12. Provinsi A I. Kota Serang \nBanten 2. Kota Cilegon \n3. Kab. Serang \n4. Kab. Pandeglang "
    },
    {
        "page": 85,
        "content": "-86-\n5. Kab.Lebak \n6. Kab. Tangerang \n13 Provinsi Jawa D l. Kota Bandung . \nBarat 2. Kota Cimahi \n3. Kab.Bandung \n4. Kab. Bandung Barat \nF l. Kota Bogor \n2. Kab. Bogor \n3. Kab. Cianjur \n4. Kab. Sukabumi \n5. Kota Sukabumi \nT l. Kab. Purwakarta \n2. Kab. Karawang \n3. Kab.Subang \nE l. Kota Cirebon \n2. Kab. Cirebon \n3. Kab. Indramayu \n4. Kab. Malajengka \n5. Kab. Kuningan \nZ l. Kab. Garut \n2. Kab. Sumedang \n3. Kota Tasikmalaya \n4. Kab. Tasikmalaya \n5. Kab. Ciamis \n6. Kota Banjar \n7. Kab. Pangandaran \n14. Provinsi H l. Kodya Semarang \nJateng 2. Kab. Salatiga \n3. Kab. Kendal \n4. Kab. Demal< \n5. Kab. Semarang \nG l. Kodya Pekalongan \n2. Kab. Pekalongan \n3. Kab. Brebes \n4. Kodya Tegal \n5. Kab. Tegal "
    },
    {
        "page": 86,
        "content": "-87-\n6. Kab. Batang \n7. Kab. Pemalang \nK 1. Kab.Pati \n2. Kab. Kudus \n3. Kab. Jepara \n4. Kab. Rembang \n5. Kab. Elora \n6. Kab. Grobogan \nR 1. Kab. Banyumas \n2. Kab. Cilacap \n3. Kab. Purbalingga \n4. Kab. Banjarnegara \nAA 1. Kodya Magelang \n2. Kab. Mage1ang \n3. Kab. Purworejo \n4. Kab. Kebumen \n5. Kab. Temanggung \n6. Kab. Wonosobo \nAD 1. Kodya Surakarta \n2. Kab. Sukoharjo \n3. Kab. Boyolali \n4. Kab. Sragen \n5. Kab. Karanganyar \n6. Kab. Wonogiri \n7. Kab. Klaten \n15. Provinsi AB 1. Kota Y ogyakarta \nDaerah 2. Kab. Bantul \nIstimewa 3. Kab. Gunung Kidul \nYogyakarta 4. Kab. Sleman \n5. Kab. Kulon Progo \n16. Provinsi Jawa L Kodya Surabaya \nTimur \nW 1. Kab. Gresik \n2. Kab. SidoaIjo \n3. Kab. Mojokerto \n4. Kab. Jombang "
    },
    {
        "page": 87,
        "content": "-88 -\nN l. Kodya Malang \n2. Kab. Malang \n3. Kab. Probolinggo \n4. Kab.Pasuruan \n5. Kab. Lumajang \np l. Kab. Besuki \n2. Kab. Situbondo \n3. Kab. Bondowoso \n4. Kab. Jember \n5. Kab. Banyuwangi \nAG 1. Kodya Kediri \n2. Kab. KedirijPare \n3. Kab. Blitar \n4. Kab. Tulungagung \n5. Kab. Nganjuk \n6. Kab. Trenggalek \nAE 1. Kodya Madiun \n2. Kab. Madiun \n3. Kab. Ngawi \n4. Kab. Magetan \n5. Kab. Ponorogo \n6. Kab. Pacitan \nS 1. Kab. Bojonegoro \n2. Kab.Tuban \n3. Kab. Lamongan \nM 1. Kab. Pamekasan \n2. Kab. Bangkalan \n3. Kab. Sampang \n4. Kab. Sumenep \n17. Provinsi Bali DK 1. Kota Denpasar \n2. Kab.Badung \n3. Kab. Buleleng \n4. Kab.Tabanan \n5. Kab. Gianyar \n6. Kab. Klungkung \n7. Kab. Bangli "
    },
    {
        "page": 88,
        "content": "-89 -\n8. Kab. Karangasem \n9. Kab. Jembrana \n18. Provinsi Nusa DR l. Kota Mataram \nTenggara 2. Kab. Lombok Barat \nBarat 3. Kab. Lombok Tengah \n4. Kab. Lombok Timur \n5. Kab. Lombok Utara \nEA 1. KotaBima \n2. Kab.Bima \n3. Kab. Sumbawa \n4. Kab. Sumbawa Barat \n5. Kab. Dompu \n19. Provinsi Nusa DH l. Kota Kupang \nTenggara 2. Kab. Timor Tengah Selatan \nTimur 3. Kab. Timor Tengah Utara \n4. Kab. Belu \n5. Kab.Kupang \n6. Kab. Sabu Raijua \n7. Kab. Rote Ndao \n8. Kab. Malaka \nEB l. Kab. Ende \n2. Kab. Sikka \n3. Kab. Flores Timur \n4. Kab.Ngada \n5. Kab. Manggarai \n6. Kab. Alor \n7. Kab. Lembata \n8. Kab. Manggarai Barat \n9. Kab. Nagekeo \n10. Kab. Manggarai Timur \nED l. Kab. Sumba Timur \n2. Kab. Sumba Barat \n3. Kab.Sumba Barat Daya \n4. Kab. Sumba Tengah "
    },
    {
        "page": 89,
        "content": "-90 -\n20. Provinsi KB 1. Kota Pontianak \nKalimatan 2. Kab. Pontianak \nBarat 3. Kab. Sambas \n4. Kab. Sanggau \n5. Kab. Sin tang \n6. Kab. Kapuas Hulu \n7. Kab. Ketapang \n8. Kab. Kubu Raya \n9. Kota Singkawang \n10. Kab.Bengkayang \nII. Kab.Landak \n12. Kab.Sekadau \n13. Kab. Melawi \n14. Kab. Kayong Utara \n21. Provinsi DA 1. Kota Banjarmasin \nKalimantan 2. Kota Banjar Baru \nSelatan 3. Kab. Balangan \n4. Kab. Banjar \n5. Kab. Barito Kuala \n6. Kab. Hulu Sungai Utara \n7. Kab. Hulu Sungai Selatan \n8. Kab. Hulu Sungai Tengah \n9. Kab. Kotabaru \n10. Kab. Tabalong \n11. Kab. Tanah Bumbu \n12. Kab. Tanah Laut \n13. Kab. Tapin \n22. Provinsi KH l. Kota Palangkaraya \nKalimantan 2. Kab. Barito Selatan \nTengah 3. Kab. Barito Timur \n4. Kab. Barito Utara \n5. Kab. Gunung Mas \n6. Kab.Kapuas \n7. Kab. Katingan \n8. Kab. Kotawaringin Barat \n9. Kab. Kotawaringin Timur "
    },
    {
        "page": 90,
        "content": "-91 -\n10. Kab. Lamandau \nII. Kab. Murung Raya \n12. Kab. Pulang Pisau \n13. Kab. Seruyan \n14. Kab. Sukamara \n23. Provinsi KT I. Kodya Balikpapan \nKalimantan 2. Kodya Samarinda \nTimur 3. Kab. Kutai Kartanegara \n4. Kab. Kutai Timur \n5. Kab. Berau \n6. Kab. Kutai Barat \n7. Kab. Mahakam UIu \n8. Kodya Bontang \n9. Kab. Paser Penajam Utara \n10. Kab. Paser \n24. Provinsi KU I. Kotamadya Tarakan \nKalimantan 2. Kab.Nunukan \nUtara 3. Kab. BuIungan \n4. Kab. Malinau \n5. Kab. Tana Tidung \n25. Provinsi DB I. KotaManado \nSulawesi 2. Kota Kotamobagu \nUtara 3. Kota Bitung \n4. Kota Tomohon \n5. Kab. BoIaang Mongondow \n6. Kab. Bolaang Mongondow Utara \n7. Kab. Bolaang Mongondow Timur \n8. Kab. Bolaang Mongondow \nSelatan \n9. Kab. Minahasa \n10. Kab. Minahasa Tenggara \nII. Kab. Minahasa Selatan \n12. Kab. Minahsa Utara \nDL 1. Kab. Sangie Talaud \n2. Kab. Kep. Talaud "
    },
    {
        "page": 91,
        "content": "-92 -\n3. Kab. Kep. Siau Tagulandang \nBiaro \n26. Provinsi DM 1. Kota Gorontalo \nGorontalo 2. Kab. Gorontalo \n3. Kab. Boalemo \n4. Kab. Pohuwato \n5. Kab. Bone Bolango \n6. Kab. Gorontalo Utara \n27. Provinsi DN 1. Kota Palu \nSulawesi 2. Kab. Banggai \nTengah 3. Kab. Banggai Kepulauan \n4. Kab. Buol \n5. Kab. Donggala \n6. Kab. Morowali \n7. Kab. Parigi Mountong \n8. Kab. Poso \n9. Kab. Tojo Una-Una \n10. Kab.Toli-Toli \n11. Kab. Sigi \n12. Kab. Banggai Laut \n28. Provinsi DD 1. Kodya Makassar \nSulawesi 2. Kab.Gowa \nSelatan 3. Kab. Takalar \n4. Kab. Maros \n5. Kab. Pangkajene kep \n6. Kab. Bantaeng \n7. Kab. Jeneponto \n8. Kab. Bulukumba \n9. Kab. Selayar \nDP 1. Kodya Pare-Pare \n2. Kab. Barru \n3. Kab. Sidrap \n4. Kab. Pinrang \n5. Kab. Palopo \n6. Kab. Luwu \n7. Kab. Luwu Timur "
    },
    {
        "page": 92,
        "content": "-93-\n8. Kab. Luwu Utara \n9. Kab. Tana Toraja \n10. Kab.Enrekang \nII. Kab. Toraja Utara \nDW I. Kab. Bone \n2. Kab. Wajo \n3. Kab.Sopeng \n4. Kab. Sinjai \n5. Kab. Watampone \n29. Provinsi DC I. Kab. Majene \nSulawesi 2. Kab. Mamuju \nBarat 3. Kab. Polewali Mandar \n4. Kab. Mamasa \n5. Kab. Mamuju Utara \n6. Kab. Mamuju Tengah \n30. Provinsi DT I. Kota Kendari \nSulawesi 2. Kota Bau-Bau \nTenggara 3. Kab. Bombana \n4. Kab. Buton \n5. Kab.Konawe \n6. Kab. Kolaka \n7. Kab. Kolaka Utara \n8. Kab. Konawe Selatan \n9. Kab. Muna \n10. Kab. Wakatobi \nII. Kab. Konawe Utara \n12. Kab. Buton Utara \n13. Kab. Kolaka Timur \n31-Provinsi DE 1-KotaAmbon \nMaluku 2. Kota Tual \n3. Kab. Buru \n4. Kab. Kepulauan Aru \n5. Kab. Seram Bagian Barat \n6. Kab. Seram Bagian Timur \n7. Kab. Maluku Tengah \n8. Kab. Maluku Tenggara "
    },
    {
        "page": 93,
        "content": "-94-\n9. Kab. Maluku Tenggara Barat \n10. Kab. Maluku Barat Daya \nII. Kab. Burn Selatan \n32. Provinsi DG I. Kota Ternate \nMaluku 2. Kota Tidore \nUtara 3. Kab. Halmahera Utara \n4. Kab. Halmahera Barat \n5. Kab. Halmahera Selatan \n6. Kab. Halmahera Tengah \n7. Kab. Halmahera Timur \n8. Kab. Pulau Morotai \n9. Kab. Kepulauan Sula \n10. Kab. Pulau Taliabu \n33. Provinsi PA I. Kab. Jayapura \nPapua 2. Kota Jayapura \n3. Kab. Jayawijaya \n4. Kab. Biak Numfor \n5. Kab. Merauke \n6. Kab. Paniai \n7. Kab.Yapen Waropen \n8. Kab. Tolikara \n9. Kab. Mimika \n10. Kab. Yahukimo \nII. Kab.Nabire \n12. Kab. Mappi \n13. Kab. Boven Digoel \n14. Kab. Asmat \n15. Kab. Sarmi \n16. Kab. Mamberamo Raya \n17. Kab. Waropen \n18. Kab. Keerom \n19. Kab. Pegunungan Bintang \n20. Kab. Puncak Jaya \n21. Kab. Supiori \n22. Kab. Mamberamo Tengah \n23. Kab. Yalimo "
    },
    {
        "page": 94,
        "content": "-95 -\n24. Kab. Lanny Jaya \n25. Kab. Nduga \n26. Kab.Puncak \n27. Kab. Dogiyai \n28. Kab. Deiyai \n29. Kab. Intan Jaya \n34. Provinsi PB l. Kab. Manokwari \nPapua Barat 2. Kab. Teluk Bintuni \n3. Kab. Teluk Wondarna \n4. Kota. Sorong \n5. Kab. Sorong \n6. Kab. Sorong Selatan \n7. Kab. Kep. Raja Ampat \n8. Kab. Fak-Fak \n9. Kab.Kaimana \n10. Kab. Tambrauw \n11. Kab. Maybrat \n12. Kab. Manokwari Selatan \n13. Kab. Pegunungan Arfak \nB. Penentuan kode registrasi adalah sebagai berikut: \nNO RANMOR KODE REGISTRASI \n1. NRKB RANMOR KORPS DIPLOMATIK CD \n2. NRKB RANMOR KORPS KONSULAT CC \n3. NRKB STNK/TNKB KHUSUS RANMOR CH KONSUL KEHORMATAN \nNRKB STNK/TNKB KHUSUS RANMOR \n4. DINAS PRESIDEN, WAKIL PRESIDEN, RI KETUA LEMBAGA TINGGI NEGARA DAN \nPEJABAT SETINGKAT MENTERI "
    },
    {
        "page": 95,
        "content": "-96-\nC. Penentuan nornor urut registrasi Ranrnor dan seri huruf. \n1) nomor urut registrasi dialokasikan sesuai jenis Ranmor yaitu: \nNO. NOMOR URUT REGISTRASI JENIS RANMOR \nl. 1 s.d. 1999 Mobil Penumpang \n2. 2000 s.d. 6999 Sepeda Motor \n3. 7000 s.d. 7999 Mobil Bus \n4. 8000 s.d. 8999 Mobil Barang \n5. 9000 s.d. 9999 Kendaraan Khusus \n2) Khusus untuk wilayah hukum Po1da Metro Jaya, nomor urut \nregistrasi dialokasikan sesuai jenis Ranmor yaitu: \nNO. NOMOR URUT REGISTRASI JENIS RANMOR \n1. 1 s.d. 2999 Mobil Penumpang \n2. 3000 s.d. 6999 Sepeda Motor \n3. 7000 s.d. 7999 Mobil Bus \n4. 8000 s.d. 8999 Mobil Penumpang \n5. 9000 s.d. 9999 Mobil barang dan Kendaraan \nKhusus; \n3) nomor urut registrasi yang digunalmn dalam satu seri huruf telah \nhabis digunakan, maka nomor urut registrasi berikutnya kembali \nke nomor awal dengan menggunakan seri huruf sesuai urutan huruf \nA sampai dengan Z: \na. untuk mobil penumpang: \nUrutan nomor Menjadi registrasi \n1 s.d. 1999 Kode Wil.-1 s.d. 1999 \n2000 s.d. 2999 Kode Wil.-1 A s.d. 1999 A \n3000 s.d. 3999 Kode Wil.-1 B s.d. 1999 B \ndan seterusnya "
    },
    {
        "page": 96,
        "content": "-97-\nb. untuk sepeda motor: \nUrutan nomor Menjadi registrasi \n1 s.d. 4999 Kode Wil.-2000 s.d. 6999 \n5000 s.d. 9999 Kode Wil.-2000 A s.d. 6999 A \n10000 s.d. 14999 Kode Wil.-2000 B s.d. 6999 B \ndan seterusnya \nc. untuk mobil bus: \nUrutan nomor Menjadi registrasi \n1 s.d. 999 Kode Wil.-7000 s.d. 7999 \n1000 s.d. 1999 Kode Wil.-7000 A s.d. 7999 A \n2000 s.d. 2999 Kode Wil.-7000 B s.d. 7999 B \ndan seterusnya \nd. untuk mobil barang: \nUrutan nomor Menjadi registrasi \n1 s.d. 999 Kode Wil.-SOOO s.d. S999 \n1000 s.d. 1999 Kode Wil.-SOOO A s.d. S999 A \n2000 s.d. 2999 Kode Wil.-SOOO B s.d. S999 B \ndan seterusnya \ne. untuk kendaraan khusus: \nUrutan nomor Menjadi registrasi \n1 s.d. 999 Kode Wil.-9000 s.d. 9999 \n1000 s.d. 1999 Kode Wil.-9000 A s.d. 9999 A \n2000 s.d. 2999 Kode Wil.-9000 B s.d. 9999 B \ndan seterusnya "
    },
    {
        "page": 97,
        "content": "-98 -\n1) apabila seri huruftelah sampai pada hurufZ, maka penomoran dapat \nmenggunakan 2(dua) seri huruf. \ncontoh: untuk Mobil Penumpang: \nUrutan nomor Menjadi registrasi \n51.974 s.d. 53.973 Kode Wil.-1 Z s.d. 1999 Z \n53.974 s.d. 55.972 Kode Wil.-1 AA s.d. 1999 AA \n55.973 s.d. 57.971 Kode Wil.-1 AB s.d. 1999 AB \ndan seterusnya \n2) apabila seri huruf telah sampai pada huruf ZZ, maka penomoran \ndapat menggunakan 3(tiga) seri huruf. \ncontoh: untuk Mobil Penumpang: \nUrutan Registrasi Menjadi \n51.974 s.d. 53.973 Kode Wil. 1 ZZ s.d. 1999 ZZ \n53.974 s.d. 55.972 Kode Wil. 1 AAA s.d. 1999 AAA \n55.973 s.d. 57.971 Kode Wil. 1 AAB s.d. 1999 AAB \ndan seterusnya \nD. NRKB untuk Ranmor PNA: \n1) NRKB untuk Ranmor Korps Diplomatik terdiri dari Kode wilayah \nmenggunakan kode registrasi \"CD\", kode negara dan nomor urut \nregistrasi; \nContoh: Ranmor Korps Diplomatik Amerika Serikat \nKode wilayahjkode Kode negara Nomorurut \nregistrasi registrasi \nCD 12 5 \n2) NRKB untuk Ranmor Korps Konsulat terdiri dari Kode wilayah \nmenggunakan kode registrasi \"CC\", kode negara dan nomor urut \nregistrasi; "
    },
    {
        "page": 98,
        "content": "-99 -\nContoh: Ranmor Korps Konsulat Amerika Serikat \nKode wilayahjkode Kode negara Nomorurut \nregistrasi registrasi \nCC 12 10 \n3) NRKB untuk Ranmor Korps Diplomatik dan Korps Konsulat, \ndikoordinasikan dengan Kementerian Luar Negeri; \n4) susunan kode negara dengan kode registrasi CD untuk Korps \ndiplomatik dan perwakilan tetap ASEAN adalah sebagai berikut: \nNO. NEGARA KODE NEGARA \nl. Amerika Serikat 12 \n2. India 13 \n3. Perancis 14 \n4. Inggris 15 \n5. Filipina 16 \n6. Vatikan 17 \n7. Australia 18 \n8. Norwegia 19 \n9. Irak 20 \n10. Pakistan 21 \nII. Belgia 22 \n12. Myanmar 23 \n13. Uni Emirat Arab 24 \n14. China 25 \n15. Swedia 26 \n16. Saudi Arabia 27 \n17. Thailand 28 \n18. Mesir 29 \n19. Italia 30 \n20. Swiss 31 \n2I. Jerman 32 \n22. Srilanka 33 \n23. Denmark 34 \n24. Kanada 35 "
    },
    {
        "page": 99,
        "content": "-100-\n25. Brazil 36 \n26. Rusia 37 \n27. Afganistan 38 \n28. Serbia 39 \n29. Ceko 40 \n30. Finlandia 41 \n31. Meksiko 42 \n32. Hungaria 43 \n33. Polandia 44 \n34. Iran 45 \n35. Malaysia 47 \n36. Turki 48 \n37. Jepang 49 \n38. Bulgaria 50 \n39. Kamboja 51 \n40. Argentina 52 \n41. Rumania 53 \n42. Yunani 54 \n43. Yordania 55 \n44. Austria 56 \n45. Syria 57 \n46. United Nations Development 58 Proaramme (UNDP) \n47. Selandia Baru 59 \n48. Belanda 60 \n49. Yaman 61 \n50. Universal Postal Union (UPU) 62 \n51. Portugal 63 \n52. Aljazair 64 \n53. Korea Utara 65 \n54. Vietnam 66 \n55. Singapura 67 \n56. Spanyol 68 \n57. Bangladesh 69 "
    },
    {
        "page": 100,
        "content": "-101 -\n58. Panama 70 \n59. United Nations International Children's 71 Emergency Fund (UNICEF) \n60. United Nations Educational, Scientific 72 and Cultural Omanization (UNESCO) \n61. Food and Agriculture Organization 73 (FAO) \n62. World Health Organization (WHO) 74 \n63. Korea Selatan 75 \n64. Asian Development Bank (ADB) 76 \n65. International Bank for Reconstruction 77 and Development (IBRD) jBank Dunia \n66. International Monetary Fund (IMF) 78 \n67. International Labour Organisation (ILO) 79 \n68. Papua Nugini 80 \n69. Nigeria 81 \n70. Chili 82 \n71. United Nations High Commissioner for 83 Refuqees (UNHCR) \n72. World Food Programme (WFP) 84 \n73. Venezuela 85 \n74. Economic and Social Commission for 86 Asia and the Pacific (ESCAP) \n75. Kolombia 87 \n76. Brunei Darussalam 88 \n77. United Nations Information Center 89 (UNIC) \n78. International Finance Corporation \n(IFC) 90 \n79. Perutusan Tetap Republik Indonesia \n(PTRI) . ASEAN 92 \n80. Fiji 93 \n81. Belarus 94 \n82. Kazakhtan 95 \n83. United Nations Industrial Development \nOr9anization (UNIDO) 96 \n84. International Committee of the Red 97 \nCross(ICRC) \n85. Maroko 98 \n86. Uni Eropa 99 \n87. Association of Southeast Asian Nations 100 (ASEAN) "
    },
    {
        "page": 101,
        "content": "-102 -\n88. Tunisia 101 \n89. Kuait 102 \n90. Laos 103 \n9I. Palestina 104 \n92. Kuba 105 \n93. ASEAN Inter-Paliamentary 106 Organization (AlPA) \n94. Libya 107 \n95. Peru 108 \n96. Slovakia 109 \n97. Sudan 110 \n98. ASEAN Foundation 111 \n99. Utusan Sekjen PBB 112 \n100. Center for International Forestry 113 Research (ClFOR\\ \n10I. Bosnia Herzegovina 114 \n102. Lebanon 115 \n103. Afrika Selatan 116 \n104. Kroasia 117 \n105. Ukraina 118 \n106. Uzbekistan 120 \n107. Qatar 121 \n108. United Nations Population Fund 122 (UNFPA) \n109. Mozambique 123 \n110. Timor Leste 125 \n111. Suriname 126 \n112. Ecuador 127 \n113. Zimbabwe 128 \n114. Perwakilan International Organization \nfor Migration (lOM) 129 \n115. Azerbaijan 130 \n116. Somalia 131 \n117. Georgia 132 \n118. Paraguay 133 \n119. Oman 134 "
    },
    {
        "page": 102,
        "content": "-103-\n120. Armenia 135 \n121. Bahrain 136 \n122. Mongolia 137 \n123. San Marino 138 \n124. Irlandia 139 \n125. United Nations Office for REDD+ 140 Coordination in Indonesia IUNORCID) \n126. Islamic Development Bank (IDB) 141 \n127. Guinea Bissau 142 \n128. Ethiopia 143 \n129. Kep Solomon 144 \n130. International Fund for Agricultural 145 Development (IFAD) \nE. NRKB untuk Ranmor Badan Internasional: \n1) NRKB Ranmor badan internasional dengan penangguhan bea masuk \n(Formulir B atau otomasi data B) terdiri dari kode wilayah sesuai \ndengan wilayah Regident Ranmor, nomor urut registrasi sesuai \ndengan jenis Ranmor dan seri huruf yang diatur oleh Kapolda. \n2) TNKB untuk Ranmor Badan Internasional disesuaikan dengan status \nkepemilikan. \nF. NRKB untuk STNK dan TNKB Khusus. \n1) STNK dan TNKB khusus dapat diberikan kepada: \na. Ranmor dinas Presiden; \nb. Ranmor dinas Wakil Presiden; \nc. Ranmor dinas Ketua Lembaga Tinggi Negara; \nd. Ranmor dinas pejabat setingkat Menteri; \ne. Ranmor dinas pejabat TNI/Polri dan instansi pemerintah eselon \nI, II dan III; dan \nf. Ranmor pejabat konsul kehormatan. \n2) NRKB untuk Ranmor Dinas Presiden, Waldl Presiden, Ketua Lembaga \nTinggi Negara dan pejabat setingkat menteri dengan kode wilayah \nmenggunakan kode Registrasi RI dan nomor urut registrasi tanpa seri \nhuruf; "
    },
    {
        "page": 103,
        "content": "-lO4-\nSusunan NRKB Ranmor Dinas Presiden Republik Indonesia: \nKode wilayah/kode Nomor urut Seri huruf registrasi registrasi \nRI 1 -\n3) nomor urut registrasi untuk Ranmor Dinas Presiden, Wakil Presiden \ndan pejabat TNI/Polri serta instansi pemerintah setingkat menteri \nsebagaimana dimaksud pada huruf b, dikoordinasikan dengan \nKementerian Sekretariat Negara; \n4) NRKB untuk Ranmor Dinas pejabat TNI/Polri serta instansi \npemerintah setingkat eselon I, II dan III, terdiri dari kode wilayah \nsesuai dengan wilayah Regident Ranmor, nomor urut registrasi dan \nseri hurufyang diatur oleh Kapolda; \nSusunan NRKB Ranmor Dinas Pejabat Polri setingkat eselon I,ll dan \nIII di wilayah Polda Metro Jaya: \nKode wilayah Nomor urut registrasi seri huruf \n---- ---B (diatur oleh Kapolda) (diatur oleh Kapolda) \n5) NRKB untuk Ranmor Pejabat Konsul kehormatan dengan kode \nwilayah menggunakan kode registrasi CH, kode negara dan nomor \nurut registrasi; \nSusunan NRKB Konsul Kehormatan Azerbaijan: \nKode wilayah/kode Kode negara Nomor urut registrasi registrasi \nCH 130 1 \n6) Nomor urut registrasi sebagaimana dimaksud pada huruf e untuk \nRanmor Konsul Kehormatan dikoordinasikan dengan Kementerian \nLuar Negeri. \nG. NRKB untuk STNK dan TNKB Rahasia. \n1) STNK dan TNKB Rahasia diberikan kepada: \na. Intelijen TNI; \nb. Intelijen Polri; \nc. Intelijen Kejaksaan; "
    },
    {
        "page": 104,
        "content": "-105-\nd. Badan Intelijen Negara; dan \ne. Penyidik/Penyelidik; \n2) Ranmor dinas yang dapat diberikan STNK dan TNKB Rahasia adalah \nRanmor model: \na. sedan; \nb. Jeep; \nc. mini bus; dan \nd. sepeda motor. \n3) NRKB untuk STNK dan TNKB Rahasia, terdiri dari kode wilayah sesuai \ndengan wilayah Regident Ranmor, nomor urut registrasi sesuai \ndenganjenis Ranmor dan seri hurufyang diatur oleh Kapolda. \n4) Ranmor dinas TNI/Polri/Pemerintah yang tidak diperbolehkan \nmenggunakan STNK dan TNKB Rahasia/Khusus: \na. Ranmor dinas POM TNI/Provos; \nb. Ranmor dinas batalyon tempur; \nc. Ranmor dinas patroli; \nd. Ranmor dinas operasional Polisi Lalu Lintas; \ne. Ranmor dinas ambulans rumah sakit; dan \nRanmor dinas pemadam kebakaran/kebersihan kota; dan \nkendaraan bermotor dinas jenis khusus. \nH. NRKB untuk Ranmor dinas jabatan bagi pejabat pemerintah provinsi \nsebagai berikut: \n1) kode wilayah, nomor urut registrasi 1, tanpa seri huruf, untuk \nGubernur; \n2) kode wilayah, nomor urut registrasi 2, tanpa seri huruf, untuk Wakil \nGubernur; \n3) kode wilayah, nomor urut registrasi 3, tanpa seri huruf, untuk Ketua \nDPRD Provinsi; \n4) kode wilayah, nomor urut registrasi 4 sampai dengan 99 tanpa seri \nhuruf, untuk pejabat lainnya sesuai urutan pejabat sipil daerah \nProvinsi masing-masing; \n1. NRKB untuk Ranmor dinas jabatan bagi pejabat pemerintah khusus \nprovinsi DKI Jakarta sebagai berikut: \n1) kode wilayah, nomor urut registrasi 1, denganseri huruf DKI, untuk \nGubernur; "
    },
    {
        "page": 105,
        "content": "-106 -\n2) kode wilayah, nomor urut registrasi 2, dengan seri huruf DKI, untuk \nWakil Gubernur; \n3) kode wilayah, nomor urut registrasi 3, dengan seri huruf DKI, untuk \nKetua DPRD Provinsi; \n4) kode wilayah, nomor urut registrasi 4 sampai dengan 150 dengan seri \nhuruf DKI, untuk Pejabat lainnya sesuai urutan pejabat sipil daerah \nProvinsi DKI Jakarta; \nJ. NRKB untuk Ranmor dinas jabatan bagi pejabat pemerintah di daerah \nkabupaten/kota, diatur sebagai berikut: \n1) kode wilayah, nomor urut registrasi 1, dengan seri huruf alokasi \nkabupaten/kota, untuk Bupati/Walikota; \n2) kode wilayah, nomor urut registrasi 2, dengan seri huruf alokasi \nkabupaten/kota, untuk Wakil Bupati/Wakil Walikota; \n3) kode wilayah, nomor urut registrasi 3, dengan seri huruf alokasi \nkabupaten/kota, untuk Ketua DPRD Kabupaten/Kota; \n4) kode wilayah, nomor urut registrasi 4 sampai dengan 30 dengan \nalokasi seri huruf awal untuk kabupaten/kota, untuk pejabat lainnya \nsesuai urutan pejabat sipil daerah kabupaten/kota masing-masing; \nK. NRKB untuk STCK dan TCKB \n1) kendaraan bermotor yang belum diregistrasi tetapi dioperasikan \ndi jalan dengan kepentingan tertentu, menggunakan nomor registrasi \nsementara yang ditandai dengan huruf seri XX, XY, YY, dan YX; \n2) penggunaan huruf seri XX, XY, YY dan YX sebagaimana dimaksud \nayat (1) diberikan kepada badan usaha di bidang penjualan, \npembuatan, perakitan, atau impor kendaraan bermotor; \n3) untuk Polda yang telah dizinkan menggunakan 3 (tiga) huruf seri, \npenerbitan nomor Registrasi sementara pada STCK apabila 2 (dua) \nhuruf seri sudah habis dapat menggunakan 3 (tiga) huruf seri XXX, \nXYY, YYX, YXY, YYY dan YXX. \nL. Susunan kode pengoperasian untuk Ranmor Asing yang dioperasikan \nsementara di wilayah Negara Republik Kesatuan Indonesia dengan STNK\u00ad\nLBN dan TNKB-LBN sebagai berikut: "
    }
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)