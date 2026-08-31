import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer



def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

model = SentenceTransformer("all-MiniLM-L6-v2") #384
text = "Machine learning is fun."

# embedding=model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1="There are 24 paid leaves"
t2="who is beast Harry"
 #in this situation t1 is very differ from t2 so -0.14....
#if t1 and t2 most similar cosine_similarity is 1 and less similarity 0.5 and most less similarity -0.5

v1=model.encode(t1)
v2=model.encode(t2)
print(cosine_similarity(v1, v2))