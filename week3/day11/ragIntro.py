import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"  




# step 1
knowledge_base={
    "age" : " The age of hareem  is 19 years",
    "net worth" : "The net worth of hareem  is 1000cr"
}

# step 2 retreieval -- so rigid software match word by word so it is not good -- seeing the questions keyword and retrieve from knowlege base ,if u write what is age still giving 25 which is not right
def retrieve_info(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return None

    
def ask_llm(question):
    context=retrieve_info(question)
    sys_prompt=f"""answer in one line only.answer only based on this context.context:{context}"""
    sys_message={
        "role":"system",
        "content":sys_prompt
    }
    message={
        "role":"user",
        "content":question
    }

    messages=[sys_message,message]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer

question="who is hareem  and what is his age"
print(ask_llm(question))