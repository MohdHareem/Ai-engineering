import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") #384

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
groqmodel="openai/gpt-oss-120b"

documents = [
    "Employees receive 24 days of paid leave per year.",
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
    "Employees receive Rs 3000 per month for gym reimbursement.",
    "Employees can claim Rs 2000 per month for home internet.",
    "Employees have a 90 day notice period.",
    "The average annual package for employees is Rs 12 LPA, "
    "including base salary and performance bonus."
]

document_embedding=model.encode(documents)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def retrieve(qembedding):
    scores=[]
    for i,document in enumerate(document_embedding):
        score=cosine_similarity(qembedding,document)
        scores.append((score,documents[i]))
    scores.sort(reverse=True)
    return scores[0]


def ask_llm(question,context):

    sys_prompt=f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=groqmodel, messages=messages)
    answer=response.choices[0].message.content
    return answer



query="what is the package of the employee "
qembedding=model.encode(query)

score,context=retrieve(qembedding)
anwser=ask_llm(query,context)
print(anwser)