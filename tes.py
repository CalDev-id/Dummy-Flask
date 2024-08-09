import requests
import os

from groq import Groq

#prepare the apiKey
with open('api_key.txt', 'r') as txt_r:
    os.environ["GROQ_API_KEY"] = txt_r.readlines()[0]

class GroqRunTime():
    def __init__(self):
        self.client = Groq(
            # this is the default and can be omitted
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def generate_response(self, system_prompt, user_prompt):
        responses = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model = "llama3-70b-8192",
            temperature = 0.3
            # repetition_penalty = 0.8,
        )
        return responses
    
if __name__ == "__main__":
    groq_run = GroqRunTime()

    system_prompt = '''
    saya ingin kamu membuat pertanyaan dari input user.
    saya ingin kamu membuat response dalam bahasa indonesia.
    pastikan pertanyaan yang kamu buat sesuai dan jawabannya ada di dalam input user.

    saat membuat response, ikutilah format dibawah ini:
    Input: [disini tempat user memasukkan input sentence].
    Pertanyaan: 
    1. [disini kamu membuat pertanyaan pertama dari input yang diberikan oleh user].
    2. [disini kamu membuat pertanyaan kedua dari input yang diberikan oleh user].
    3. [disini kamu membuat pertanyaan ke-n dari input yang diberikan oleh user].
    '''

    user_prompt = '''
    bahwa untuk melaksanakan ketentuan pasal 64 ayat (6),
    pasal 68 ayat (6), pasal 69 ayat (3), pasal 72 ayat (3)
    dan pasal 75 undang undang nomor 22 tahun 2009 tentang lalu lintas dan angkutan jalan,
    telah dikeluarkan  peraturan kepala kepolisian republik indonesia nomor 5 tahun 2012 tentang registrasi dan identifikasi kendaraan bermotor.
    '''

    response = groq_run.generate_response(system_prompt, user_prompt)

    print(response.choices[0].message.content)