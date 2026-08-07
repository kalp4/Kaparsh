import io
import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
CORS(app)

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Server Configuration Error: GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=api_key)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "EduCoPilot Flask Backend is Live and Ready!"})


@app.route("/api/analyze", methods=["POST"])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({"detail": "No file uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith(".pdf"):
        return jsonify({"detail": "Only PDF files are supported."}), 400
        
    try:
        reader = PdfReader(file)
        text = ""
        
        for page in reader.pages[:25]:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        if not text.strip():
            return jsonify({"detail": "Could not extract text."}), 400
            
        client = get_gemini_client()
        
        prompt = (
            "You are an AI study assistant. Read the following text and identify all the distinct, primary concepts and topics covered in the material. "
            "Do NOT limit yourself to a specific number. Extract as many or as few core topics as naturally exist in the text. "
            "Return ONLY a JSON object with this exact structure (no extra markdown):\n"
            "{\n"
            '  "topics": [\n'
            '    {"title": "Exact Topic Name"}\n'
            "  ]\n"
            "}\n\n"
            f"Text:\n{text[:30000]}"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        
        parsed_data = json.loads(response.text)
        
        return jsonify({
            "topics": parsed_data.get("topics", []),
            "extracted_text": text[:30000]
        })
        
    except Exception as e:
        return jsonify({"detail": f"PDF Analysis failed: {str(e)}"}), 500


@app.route("/api/topic", methods=["POST"])
def get_topic_details():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        prompt = (
            f"You are an expert tutor. Using the provided text, extract detailed study materials for the topic: '{data.get('topic')}'.\n"
            "Categorize the information strictly into definitions, formulas, derivations, and general notes.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "priority": "High",\n'
            '  "definitions": [{"term": "Exact Term", "definition": "Clear definition"}], (Leave empty [] if none exist)\n'
            '  "formulas": [{"equation": "E=mc^2", "meaning": "Mass-energy equivalence"}], (Leave empty [] if none exist)\n'
            '  "derivations": [{"title": "Derivation Name", "content": "The mathematical or logical steps"}], (Leave empty [] if none exist)\n'
            '  "notes": ["Detailed point 1", "Detailed point 2"], (General bullet points)\n'
            '  "analogy": "A simple, highly effective real-world analogy explaining this concept to a beginner."\n'
            "}\n\n"
            f"Text:\n{data.get('text')}"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Topic detailing failed: {str(e)}"}), 500


@app.route("/api/schedule", methods=["POST"])
def generate_schedule():
    try:
        data = request.get_json()
        client = get_gemini_client()
        topics_json = json.dumps(data.get('topics', []))
        
        prompt = (
            f"You are an expert study planner utilizing the Spaced Repetition algorithm. Create a schedule.\n"
            f"Exam Date: {data.get('exam_date')}\n"
            f"Daily Hours: {data.get('study_hours')}\n"
            f"Topics to map out: {topics_json}\n\n"
            "Distribute them using the 'Learn, Recall, Master' spacing method. "
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "schedule": [\n'
            '    {\n'
            '      "day": 1, \n'
            '      "date": "YYYY-MM-DD", \n'
            '      "focus_area": "Initial Learning vs Active Recall",\n'
            '      "topics_to_study": ["Topic Name 1"], \n'
            '      "hours_allocated": 2.5, \n'
            '      "actionable_advice": "Specific study technique to use today"\n'
            '    }\n'
            "  ]\n"
            "}"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
        )
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Schedule generation failed: {str(e)}"}), 500


@app.route("/api/quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        client = get_gemini_client()
        topics_json = json.dumps(data.get('topics', []))
        
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
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
        )
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Quiz generation failed: {str(e)}"}), 500