import json
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

model = "openai/gpt-oss-120b"

app=FastAPI()

# allow the frontend (running on a different port/file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev ke liye sab allow, baad me apni frontend URL daal dena
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#parse resume
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []
resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question: str

def ask_candidate_stream(question: str, resume: Resume):

    system_prompt = f"""
You are Harry, an AI assistant built by the candidate below to represent
him in conversations, including mock interviews.

Here is everything you know about the candidate from their resume:

{resume.model_dump_json(indent=2)}

Rules:

1. If the question is about the candidate's background, skills, projects,
   experience, education, or career — answer strictly using the resume
   information above. Never invent facts that aren't in it. If the resume
   truly has nothing relevant, say so honestly, e.g.
   "That's not something in my resume, but here's what I can tell you..."
   and then help however you can.

2. If the question is general knowledge, technical, or not really about the
   candidate specifically (e.g. "what is Docker", "explain REST APIs",
   "is Hareem a good fit for a cloud engineer role") — answer normally and
   helpfully like a knowledgeable AI assistant, and where relevant connect
   it back to the candidate's actual skills/projects from the resume.

3. Speak in first person as the candidate when discussing his profile
   ("I have experience with...", "I built..."), and in a professional,
   confident tone — as if HR is interviewing him.

4. Never hallucinate specific facts (companies, dates, numbers) about the
   candidate. General/technical knowledge outside the resume can be
   answered freely.

5. KEEP IT SHORT. This is a spoken conversation, not a written report.
   Answer in 2-4 short sentences by default, like you're actually talking
   in an interview — not reading out a resume. Never use markdown
   (no headers, no bullet points, no bold **text**, no numbered lists).
   Only go longer than 4-5 sentences if the person explicitly asks for
   detail, a full breakdown, or "tell me more."
"""

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system={
        "role" : "system",
        "content" : system_prompt
    }
    message_user={
        "role" : "user",
        "content" : user_prompt
    }
    messages=[message_system, message_user]
    response_format={
        "type": "json_object"
    }
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume

# pdf extraction
def read_pdf(file_path: Path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


@app.get("/")
def home():
    # resume_text=read_pdf(Path("resume.pdf"))
    # resume=parse_resume(resume_text)
    # print(resume.model_dump_json(indent=2))
    # print(resume_text)
    return{
     "message":"resume Parsed"
    }

@app.get("/resume")
def download_resume():
    return FileResponse(
        path="resume.pdf",
        filename="Mohd_Hareem_Resume.pdf",
        media_type="application/pdf"
    )

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    transcription = client.audio.transcriptions.create(
        file=(audio.filename, audio_bytes),
        model="whisper-large-v3",
    )
    return {"text": transcription.text}

@app.post("/chat")
def chat(request: ChatRequest):
    resume_text = read_pdf(Path("resume.pdf"))
    resume = parse_resume(resume_text)

    def generate():
        for chunk in ask_candidate_stream(request.question, resume):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

