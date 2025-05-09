import time
from google import genai
from google.genai import types

class APIManager:
    def __init__(self):
        self.client=None
        self.model_name=None
        self.api_name=None

    def setting(self,api_name):
        if api_name=="Gemini":
            self.api_name="Gemini"
            self.gemini_setting()

    def get_key(self,api_name):
        file=open(f"AllKeys/{api_name}")
        api_key=file.readline()
        return api_key

    def gemini_setting(self):
        GEMINI_API_KEY = self.get_key("Gemini")
        self.client=genai.Client(api_key=GEMINI_API_KEY)
        self.model_name="gemini-2.0-flash"

    def get_response(self,question,instruction=None,temp=1.0):
        if self.client==None:
            return "no model is set"
        elif self.api_name=="Gemini":
            response=self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    temperature=temp),
                contents=question
            )
            time.sleep(5)
            return response.text

    def get_dialogue(self,message):
        chat=self.client.chats.create(model=self.model_name)
        response=chat.send_message(message)
        return response.text

    def token_num(self,sentence):
        total_tokens=self.client.models.count_tokens(
            model=self.model_name,
            contents=sentence
            )
        return total_tokens