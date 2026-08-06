import io
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="EduCoPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScheduleRequest(BaseModel):
    topics: list
    exam_date: str
    study_hours: float

class QuizRequest(BaseModel):
    topics: list

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server Configuration Error: GEMINI_API_KEY environment variable is missing.")
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini Client: {str(e)}")


@app.post("/api/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        
        for page in reader.pages[:15]:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text.")
            
        client = get_gemini_client()
        
        # UPGRADE: Detailed Notes & Formula Extraction
        prompt = (
            "You are an expert cognitive learning scientist. Analyze the following educational text. "
            "Extract the core topics. For each topic, provide detailed bullet-point notes summarizing the concepts, "
            "extract any mathematical, scientific, or logical formulas (if none exist, return an empty array []), "
            "determine its importance (High, Medium, Low), and generate one Active Recall Flashcard.\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "topics": [\n'
            '    {\n'
            '      "topic": "Topic Name",\n'
            '      "priority": "High/Medium/Low",\n'
            '      "notes": ["Key point 1", "Key point 2"],\n'
            '      "formulas": [{"equation": "F = ma", "meaning": "Newton\'s Second Law"}],\n'
            '      "flashcard": {"q": "Question?", "a": "Answer."}\n'
            '    }\n'
            "  ]\n"
            "}\n\n"
            f"Text to analyze:\n{text[:25000]}"
        )
        
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Analysis failed: {str(e)}")


@app.post("/api/schedule")
async def generate_schedule(req: ScheduleRequest):
    try:
        client = get_gemini_client()
        topics_json = json.dumps(req.topics)
        
        prompt = (
            f"You are an expert study planner utilizing the Spaced Repetition algorithm. Create a schedule.\n"
            f"Exam Date: {req.exam_date}\n"
            f"Daily Hours: {req.study_hours}\n"
            f"Topics: {topics_json}\n\n"
            "Do not just list topics. Distribute them using the 'Learn, Recall, Master' spacing method. "
            "High priority topics MUST appear on multiple days. "
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "schedule": [\n'
            '    {\n'
            '      "day": 1, \n'
            '      "date": "YYYY-MM-DD", \n'
            '      "focus_area": "Initial Learning vs Active Recall",\n'
            '      "topics_to_study": ["Topic 1"], \n'
            '      "hours_allocated": 2.5, \n'
            '      "actionable_advice": "Specific study technique to use today"\n'
            '    }\n'
            "  ]\n"
            "}"
        )
        
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
        )
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {str(e)}")


@app.post("/api/quiz")
async def generate_quiz(req: QuizRequest):
    try:
        client = get_gemini_client()
        topics_json = json.dumps(req.topics)
        
        prompt = (
            "You are a strict examiner. Create a 5-question multiple-choice practice exam based on these topics.\n"
            f"Topics: {topics_json}\n\n"
            "The questions must be highly challenging. Do not just test definitions; test application. "
            "For the explanation, you MUST explain exactly why the correct answer is right AND why a student might be tricked by the distractors (incorrect options).\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "quiz": [\n'
            '    {\n'
            '      "question": "Highly challenging question...",\n'
            '      "options": ["A", "B", "C", "D"],\n'
            '      "correct_answer": "Exact string of correct option",\n'
            '      "explanation": "Why correct is right, and why the distractors are common traps."\n'
            '    }\n'
            "  ]\n"
            "}"
        )
        
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
        )
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
public_dir = os.path.join(project_root, "public")

if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")