import csv
from API import APIManager
class MakeDiff:
    def __init__(self,csv_data):
        self.api_manager=APIManager()
        self.api_manager.setting("Gemini")
        self.csv_reader=csv.reader(csv_data)

    def diff_temp(self,temp):
        answers=[]
        for row in self.csv_reader:
            question=row[0]
            choice1=row[1]
            choice2=row[2]
            choice3=row[3]
            choice4=row[4]
            answer=row[5]
            prompt=f"""
        問題：{question}
        選択肢
        A:{choice1}
        B:{choice2}
        C:{choice3}
        D:{choice4}
        """
            instruction="""出力は答えのみ。1文字で答えてください。"""
            response=self.api_manager.get_response(prompt,instruction,0.5)
            answers.append((answer,response.replace("\n","")))
        return answers

    def diff_prompt(self):
        answers=[]
        for row in self.csv_reader:
            question=row[0]
            choice1=row[1]
            choice2=row[2]
            choice3=row[3]
            choice4=row[4]
            answer=row[5]
            prompt=f"""
        以下の問題は雑学に関するものです。
        問題：{question}
        選択肢
        A:{choice1}
        B:{choice2}
        C:{choice3}
        D:{choice4}
        """
            instruction="""出力は答えのみ。1文字で答えてください。"""
            response=self.api_manager.get_response(prompt,instruction)
            answers.append((answer,response.replace("\n","")))
        return answers

    def all_answer_d(self):
        answers=[]
        for row in self.csv_reader:
            question=row[0]
            choice1=row[1]
            choice2=row[2]
            choice3=row[3]
            choice4=row[4]
            answer=row[5]
            if answer=="A":
                choice4=row[1]
                choice1=row[4]
            elif answer=="B":
                choice4=row[2]
                choice2=row[4]
            elif answer=="C":
                choice4=row[3]
                choice3=row[4]
            prompt=f"""
        以下の問題は雑学に関するものです。
        問題：{question}
        選択肢
        A:{choice1}
        B:{choice2}
        C:{choice3}
        D:{choice4}
        """
            instruction="""出力は答えのみ。1文字で答えてください。"""
            response=self.api_manager.get_response(prompt,instruction)
            answers.append((answer,response.replace("\n","")))
        return answers