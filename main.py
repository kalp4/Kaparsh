import io
import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
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
    return jsonify({"status": "Kaparsh Flask Backend is Live and Ready!"})


@app.route("/api/analyze", methods=["POST"])
def analyze_pdf():
    try:
        text = ""
        client = get_gemini_client()
        
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded."}), 400
            
        file = request.files['file']
        filename = file.filename.lower()
        
        if not (filename.endswith(".pdf") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            return jsonify({"detail": "Only PDF and Image files (.pdf, .png, .jpg, .jpeg) are supported."}), 400
            
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages[:25]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            image_bytes = file.read()
            mime_type = "image/png" if filename.endswith(".png") else "image/jpeg"
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            
            transcription_response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=[
                    "You are an OCR system. Transcribe all readable text in this image perfectly without commentary.", 
                    image_part
                ]
            )
            text = transcription_response.text
            
        if not text or not text.strip():
            return jsonify({"detail": "Could not extract text. The document might be completely blank or unreadable."}), 400
            
        prompt = (
            "You are an AI study assistant. Read the following text and identify all distinct, primary concepts.\n"
            "Ensure there are NO duplicate or heavily overlapping topics. Merge similar concepts into a single title.\n"
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
        return jsonify({"detail": f"File Analysis failed: {str(e)}"}), 500


@app.route("/api/parse-syllabus", methods=["POST"])
def parse_syllabus():
    try:
        client = get_gemini_client()
        
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded."}), 400
            
        file = request.files['file']
        filename = file.filename.lower()
        
        if not (filename.endswith(".pdf") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            return jsonify({"detail": "Only PDF and Image files (.pdf, .png, .jpg, .jpeg) are supported."}), 400
            
        text = ""
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages[:20]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            image_bytes = file.read()
            mime_type = "image/png" if filename.endswith(".png") else "image/jpeg"
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            transcription_response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=["Transcribe readable text from this syllabus image perfectly. Raw text only.", image_part]
            )
            text = transcription_response.text
                
        if not text.strip():
            return jsonify({"detail": "Could not extract text from the syllabus."}), 400
            
        prompt = (
            "Analyze the following syllabus text and extract a clean list of all distinct subjects or modules found in it.\n"
            "Return ONLY a JSON object with this exact structure (no extra markdown):\n"
            "{\n"
            '  "subjects": ["Subject 1", "Subject 2", "Subject 3"]\n'
            "}\n\n"
            f"Text:\n{text[:30000]}"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        
        parsed = json.loads(response.text)
        subjects = parsed.get("subjects", [])
        if not subjects:
            subjects = ["General Coursework / Core Syllabus"]

        return jsonify({
            "subjects": subjects,
            "full_text": text
        })
        
    except Exception as e:
        return jsonify({"detail": f"Syllabus parsing failed: {str(e)}"}), 500


@app.route("/api/topic", methods=["POST"])
def get_topic_details():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        topic_name = data.get('topic')
        covered_topics = data.get('covered_topics', '')
        
        ignore_prompt = f"GLOBAL BAN LIST: The following concepts have ALREADY been defined: [{covered_topics}]. DO NOT DEFINE THEM AGAIN.\n" if covered_topics else ""
        
        prompt = (
            f"Extract detailed study materials EXCLUSIVELY for the specific topic: '{topic_name}'.\n"
            f"{ignore_prompt}"
            "Categorize information strictly into definitions, formulas, derivations, and general notes.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. MUTUALLY EXCLUSIVE: If a fact is a 'definition', do not repeat it in 'notes'.\n"
            "2. BE HIGHLY CONCISE: Strip out filler words.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "priority": "High",\n'
            '  "definitions": [{"term": "Exact Term", "definition": "Clear definition"}],\n'
            '  "formulas": [{"equation": "E=mc^2", "meaning": "Meaning"}],\n'
            '  "derivations": [{"title": "Derivation Name", "content": "Steps"}],\n'
            '  "notes": ["Concise point 1", "Concise point 2"]\n'
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
        
        exam_date = data.get('exam_date', '')
        study_hours = data.get('study_hours', 2)
        selected_subjects = data.get('selected_subjects', [])
        syllabus_text = data.get('syllabus_text', '')
        topics = data.get('topics', [])
        
        if topics:
            topics_json = json.dumps(topics)
            prompt = (
                f"You are a study planner utilizing Spaced Repetition.\n"
                f"Exam Date: {exam_date}\n"
                f"Daily Hours Available: {study_hours}\n"
                f"Topics to map out: {topics_json}\n\n"
                "Return ONLY a JSON object with this exact structure:\n"
                "{\n"
                '  "schedule": [\n'
                '    {\n'
                '      "day": 1, \n'
                '      "date": "YYYY-MM-DD", \n'
                '      "focus_area": "Active Recall",\n'
                '      "topics": [{"name": "Topic 1", "estimated_minutes": 30}], \n'
                '      "total_hours_today": 0.75, \n'
                '      "actionable_advice": "Advice"\n'
                '    }\n'
                "  ]\n"
                "}"
            )
        else:
            subjects_str = ", ".join(selected_subjects) if selected_subjects else "All Selected Subjects"
            prompt = (
                f"You are a Master Study Planner. Build a custom study schedule based on the syllabus text below.\n"
                f"Target Exam Date: {exam_date}\n"
                f"Daily Study Hours Available: {study_hours}\n"
                f"Strictly limit the schedule ONLY to these selected subjects: [{subjects_str}]. Completely ignore any other subjects or modules mentioned in the syllabus text.\n\n"
                f"Syllabus Content:\n{syllabus_text[:30000]}\n\n"
                "Return ONLY a JSON object with this exact structure:\n"
                "{\n"
                '  "schedule": [\n'
                '    {\n'
                '      "day": 1, \n'
                '      "date": "YYYY-MM-DD", \n'
                '      "focus_area": "Subject Name",\n'
                '      "topics": [{"name": "Specific Topic", "estimated_minutes": 45}], \n'
                '      "total_hours_today": 2.5, \n'
                '      "actionable_advice": "Advice"\n'
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
            "Create a multiple-choice practice exam based on these topics.\n"
            f"Topics: {topics_json}\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "quiz": [\n'
            '    {\n'
            '      "question": "Question...",\n'
            '      "options": ["A", "B", "C", "D"],\n'
            '      "correct_answer": "Exact correct option string",\n'
            '      "explanation": "Why correct is right, and distractors are wrong."\n'
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


@app.route("/api/doubt", methods=["POST"])
def answer_doubt():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        prompt = (
            "You are a helpful expert tutor. A student has a doubt regarding their study material.\n\n"
            f"Study Material Context:\n{data.get('text')}\n\n"
            f"Student's Doubt: {data.get('question')}\n\n"
            "Provide a clear, concise, and accurate explanation based ONLY on the context provided."
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )
        
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"detail": f"Failed to answer doubt: {str(e)}"}), 500


# ==============================================================================
# RESPONSIVE WEB APP UI (DESKTOP & MOBILE)
# GEMINI-INSPIRED AMBIENT BLACK AESTHETICS & PREMIUM GLASS BUTTONS
# ==============================================================================

KAPARSH_FRONTEND = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Kaparsh Intelligence</title>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

    <style>
        /* PREMIUM AMBIENT DESIGN SYSTEM (WEB APP) */
        :root {
            --bg-color: #000000;
            --surface-glass: rgba(20, 20, 20, 0.4);
            --surface-border: rgba(255, 255, 255, 0.1);
            
            /* Aurora Accents */
            --accent-primary: #8B5CF6; /* Royal Violet */
            --accent-secondary: #3B82F6; /* Bright Blue */
            --accent-tertiary: #10B981; /* Emerald */
            --danger: #EF4444;
            
            --text-primary: #F8FAFC;
            --text-secondary: rgba(255, 255, 255, 0.65);
            --text-tertiary: rgba(255, 255, 255, 0.4);
            --bezier: cubic-bezier(0.16, 1, 0.3, 1);
        }

        * {
            margin: 0; padding: 0; box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif;
        }

        body, html {
            background-color: var(--bg-color);
            color: var(--text-primary);
            width: 100%; height: 100%;
            overflow-x: hidden; scroll-behavior: smooth;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        /* Gemini-style Deep Fluid Aurora Background */
        .ambient-bg {
            position: fixed; inset: 0; z-index: -1;
            overflow: hidden; pointer-events: none;
            background: #000000;
        }
        .orb {
            position: absolute; border-radius: 50%; filter: blur(140px); opacity: 0.35;
            animation: drift 30s infinite alternate cubic-bezier(0.4, 0, 0.2, 1);
            mix-blend-mode: screen;
        }
        .orb.one { width: 90vw; height: 90vw; max-width: 1200px; background: #1D4ED8; top: -20%; left: -10%; animation-duration: 25s; }
        .orb.two { width: 80vw; height: 80vw; max-width: 1000px; background: #6D28D9; bottom: -10%; right: -20%; animation-delay: -10s; }
        .orb.three { width: 70vw; height: 70vw; max-width: 800px; background: #0891B2; top: 40%; left: 30%; opacity: 0.2; animation-delay: -15s; }
        
        @keyframes drift {
            0% { transform: translate(0, 0) scale(1) rotate(0deg); }
            50% { transform: translate(10vw, 15vh) scale(1.1) rotate(10deg); }
            100% { transform: translate(-5vw, 5vh) scale(1) rotate(-5deg); }
        }

        /* Fully Responsive Centered Web Container */
        .app-container {
            width: 100%;
            max-width: 900px; /* Expands nicely on desktop */
            margin: 0 auto;
            min-height: 100vh;
            position: relative;
            padding: 24px 20px calc(140px + env(safe-area-inset-bottom)) 20px;
            display: flex; flex-direction: column;
        }
        
        /* Header */
        .header {
            position: sticky; top: 0; z-index: 40;
            padding: 16px 0; margin-bottom: 24px;
            display: flex; justify-content: space-between; align-items: center;
            background: linear-gradient(180deg, #000000 30%, transparent);
        }
        .header h1 { 
            font-size: 24px; font-weight: 700; letter-spacing: -0.5px; 
            color: #fff;
        }

        .icon-btn {
            width: 40px; height: 40px; border-radius: 50%; border: none;
            background: rgba(255, 255, 255, 0.05); color: var(--text-primary);
            display: flex; align-items: center; justify-content: center; font-size: 16px;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.08); cursor: pointer;
            transition: all 0.3s var(--bezier); box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .icon-btn:hover { background: rgba(255,255,255,0.1); transform: translateY(-2px); }
        .icon-btn:active { transform: scale(0.9); }

        /* Typography */
        .large-title { font-size: 36px; font-weight: 700; letter-spacing: -1px; margin-bottom: 12px; line-height: 1.1; color: #fff; }
        .sub-title { font-size: 16px; color: var(--text-secondary); line-height: 1.5; font-weight: 400; margin-bottom: 28px; }
        .section-header { font-size: 20px; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; color: #FFF; }

        .hidden { display: none !important; }
        .flex { display: flex !important; }
        .tab-pane { flex: 1; animation: fadeScale 0.4s var(--bezier); display: flex; flex-direction: column; width: 100%; }
        
        @keyframes fadeScale {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glass Surface Modules */
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(30px) saturate(150%); -webkit-backdrop-filter: blur(30px) saturate(150%);
            border: 1px solid var(--surface-border);
            border-radius: 20px; padding: 28px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
            margin-bottom: 20px; position: relative; overflow: hidden;
        }

        /* Upload Zones */
        .upload-zone {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px 20px; border-radius: 20px;
            background: rgba(255, 255, 255, 0.02);
            border: 1.5px dashed rgba(255, 255, 255, 0.15);
            cursor: pointer; transition: all 0.3s var(--bezier);
            text-align: center; margin-bottom: 24px; width: 100%;
        }
        .upload-zone:hover { 
            background: rgba(255, 255, 255, 0.05); 
            border-color: var(--accent-primary); 
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.1);
        }
        .upload-zone input[type="file"] { position: absolute; opacity: 0; width: 0; height: 0; }
        .upload-icons { display: flex; gap: 16px; margin-bottom: 16px; }
        .upload-icons i { font-size: 32px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.5)); }
        .upload-text { font-size: 16px; font-weight: 500; color: #FFF; }

        /* ====== Refined Premium Glass Buttons ====== */
        .btn-primary {
            width: 100%; padding: 18px; border-radius: 18px;
            background: rgba(255, 255, 255, 0.05); color: #fff;
            border: 1px solid rgba(255,255,255,0.15); font-size: 16px; font-weight: 600; letter-spacing: 0.5px;
            cursor: pointer; transition: all 0.3s var(--bezier);
            display: flex; justify-content: center; align-items: center; gap: 10px;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            position: relative; overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        /* Beautiful hover state for desktop */
        .btn-primary:hover {
            background: rgba(255, 255, 255, 0.1); border-color: rgba(255,255,255,0.3);
            transform: translateY(-2px); box-shadow: 0 8px 30px rgba(139, 92, 246, 0.3);
        }
        .btn-primary:active { transform: scale(0.97); }
        .btn-success:hover { box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3); }

        /* Input Elements */
        .input-group {
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px; padding: 16px 20px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
        }
        .input-group label { font-size: 15px; color: var(--text-secondary); font-weight: 500; }
        .input-group input { 
            background: transparent; border: none; outline: none; color: #fff; 
            font-size: 16px; font-weight: 600; text-align: right; 
            -webkit-appearance: none; appearance: none;
        }
        .input-group input[type="date"] { color: var(--accent-tertiary); color-scheme: dark; }

        .custom-checkbox {
            display: flex; align-items: center; gap: 16px; padding: 16px 20px;
            background: rgba(255,255,255,0.03); border: 1px solid var(--surface-border);
            border-radius: 16px; transition: 0.2s var(--bezier); cursor: pointer; margin-bottom: 10px;
        }
        .custom-checkbox:hover { background: rgba(255,255,255,0.06); }
        .custom-checkbox input { display: none; }
        .checker {
            width: 24px; height: 24px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.3);
            display: flex; align-items: center; justify-content: center; transition: 0.3s var(--bezier);
        }
        .checker i { font-size: 12px; color: #fff; opacity: 0; transform: scale(0.5); transition: 0.3s var(--bezier); }
        .custom-checkbox input:checked + .checker { background: var(--accent-primary); border-color: var(--accent-primary); box-shadow: 0 0 12px var(--accent-primary); }
        .custom-checkbox input:checked + .checker i { opacity: 1; transform: scale(1); }
        .checkbox-label { font-size: 15px; font-weight: 500; flex: 1; color: #fff; }

        /* Content Styling */
        .md-content { line-height: 1.7; word-wrap: break-word; color: #E2E8F0; }
        .md-content p { margin-bottom: 14px; }
        .md-content p:last-child { margin-bottom: 0; }
        .md-content strong { color: #fff; font-weight: 600; }
        .md-content em { color: #94A3B8; }
        .md-content ul, .md-content ol { padding-left: 20px; margin-bottom: 14px; }
        .md-content li { margin-bottom: 8px; }
        .md-content code { background: rgba(255,255,255,0.08); padding: 3px 6px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 0.9em; border: 1px solid rgba(255,255,255,0.05); }
        .md-content pre { background: rgba(0,0,0,0.6); padding: 16px; border-radius: 12px; overflow-x: auto; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.1); }
        .md-content pre code { background: transparent; padding: 0; border: none; font-size: 13px; }
        .md-content h3, .md-content h4 { color: #fff; font-weight: 600; margin-top: 20px; margin-bottom: 12px; }
        
        .katex { color: #FFF; font-size: 1.05em; } 
        .katex-display { margin: 16px 0; overflow-x: auto; overflow-y: hidden; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center; }

        /* Grid Layouts for Desktop */
        @media (min-width: 768px) {
            #quiz-questions-container { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
            .q-card { margin-bottom: 0 !important; }
        }

        /* Notes Display */
        .badge {
            font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
            padding: 6px 12px; border-radius: 20px;
            background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); color: #FFF;
        }
        .note-card { border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 28px; margin-bottom: 28px; }
        .note-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .note-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; }
        .note-title { font-size: 20px; font-weight: 600; color: #fff; line-height: 1.3; }
        
        .def-box {
            background: rgba(139, 92, 246, 0.1); border-left: 3px solid var(--accent-primary);
            padding: 14px 16px; border-radius: 0 12px 12px 0; margin-bottom: 16px;
        }
        .def-box span.term-title { font-weight: 600; color: #C4B5FD; margin-right: 8px; font-size: 15px; }

        .formula-box {
            background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
            padding: 16px; border-radius: 12px; text-align: center; margin-bottom: 16px;
        }
        .formula-box .eq { font-size: 18px; color: #6EE7B7; margin-bottom: 6px; }

        /* Refined Quiz UI Fixes */
        .q-card { border: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px; padding: 24px; }
        .q-label { font-size: 13px; font-weight: 700; color: var(--accent-secondary); letter-spacing: 1px; margin-bottom: 12px; }
        .q-text { font-size: 17px; font-weight: 500; margin-bottom: 24px; }
        
        .quiz-opt-label {
            display: flex; align-items: flex-start; gap: 14px; padding: 14px 16px; margin-bottom: 12px;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px; font-size: 15px; cursor: pointer; transition: all 0.2s var(--bezier);
        }
        /* Overrides markdown generated <p> inside labels to keep it inline and tight */
        .quiz-opt-label .md-content p { margin: 0; padding: 0; display: inline; }
        
        .quiz-opt-label input { display: none; }
        .radio-indicator { width: 22px; height: 22px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.3); flex-shrink: 0; margin-top: 1px; transition: 0.2s; }
        
        .quiz-opt-label:hover { background: rgba(255,255,255,0.07); }
        .quiz-opt-label:has(input:checked) { background: rgba(139, 92, 246, 0.15); border-color: var(--accent-primary); }
        .quiz-opt-label:has(input:checked) .radio-indicator { border-color: var(--accent-primary); border-width: 6px; background: #fff; }
        
        .quiz-res-box { padding: 16px; border-radius: 12px; margin-top: 20px; font-size: 15px; }
        .quiz-res-box.correct { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }
        .quiz-res-box.wrong { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Desktop Chat Optimization */
        .chat-pane { display: flex; flex-direction: column; overflow: hidden; padding-bottom: 60px; min-height: 60vh; }
        .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-bottom: 30px; }
        .chat-bubble { max-width: 80%; padding: 16px 20px; border-radius: 18px; font-size: 15px; animation: fadeScale 0.3s var(--bezier); }
        
        .chat-ai {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
            align-self: flex-start; border-bottom-left-radius: 4px; 
            backdrop-filter: blur(20px);
        }
        .chat-user { 
            background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
            align-self: flex-end; border-bottom-right-radius: 4px; 
            backdrop-filter: blur(20px);
        }

        /* Centralized Floating Dock (Desktop + Mobile) */
        .dynamic-island {
            position: fixed; bottom: calc(20px + env(safe-area-inset-bottom)); 
            left: 50%; transform: translateX(-50%);
            background: rgba(15, 15, 15, 0.7); backdrop-filter: blur(40px) saturate(200%); -webkit-backdrop-filter: blur(40px) saturate(200%);
            border: 1px solid rgba(255,255,255,0.12); border-radius: 100px;
            display: flex; padding: 8px; gap: 4px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.6);
            z-index: 100; max-width: max-content;
        }
        .nav-indicator {
            position: absolute; top: 8px; left: 8px; width: 52px; height: 52px;
            background: rgba(255,255,255,0.15); border-radius: 50%;
            transition: transform 0.4s var(--bezier); z-index: 0; pointer-events: none;
        }
        .nav-btn {
            width: 52px; height: 52px;
            border-radius: 50%; border: none; background: transparent;
            color: rgba(255,255,255,0.5); display: flex; align-items: center; justify-content: center;
            font-size: 20px; cursor: pointer; position: relative; z-index: 1;
            transition: all 0.3s;
        }
        .nav-btn:hover { color: #fff; }
        .nav-btn.nav-active { color: #fff; transform: scale(1.05); }

        /* Floating Input Bar inside Container */
        .chat-input-wrapper {
            position: sticky; bottom: 0px; 
            width: 100%; z-index: 40; padding-bottom: 24px;
            background: linear-gradient(0deg, var(--bg-color) 40%, transparent);
        }
        .chat-input-box {
            background: rgba(20, 20, 20, 0.8); backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
            border: 1px solid rgba(255,255,255,0.15); border-radius: 30px;
            display: flex; align-items: center; padding: 8px 8px 8px 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .chat-input-box input {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 16px; width: 100%;
        }
        .chat-send-btn {
            width: 40px; height: 40px; border-radius: 50%; background: #fff;
            border: none; color: #000; display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: 0.2s; font-size: 16px; margin-left: 10px;
        }
        .chat-send-btn:hover { background: #E2E8F0; }
        .chat-send-btn:active { transform: scale(0.9); }

        /* Loader */
        .loader-overlay {
            position: fixed; inset: 0; z-index: 9999;
            background: rgba(0,0,0,0.8); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            opacity: 0; pointer-events: none; transition: opacity 0.4s ease;
        }
        .loader-overlay.active { opacity: 1; pointer-events: auto; }
        .spinner {
            width: 48px; height: 48px; border-radius: 50%;
            border: 4px solid rgba(255,255,255,0.1);
            border-top-color: #fff; animation: spin 1s infinite cubic-bezier(0.5, 0, 0.5, 1);
            margin-bottom: 24px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <div class="ambient-bg">
        <div class="orb one"></div>
        <div class="orb two"></div>
        <div class="orb three"></div>
    </div>

    <!-- Global Loader -->
    <div id="global-loader" class="loader-overlay hidden">
        <div class="spinner"></div>
        <div id="loader-title" style="font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 8px;">Processing Document...</div>
        <div id="loader-text" style="font-size: 14px; color: rgba(255,255,255,0.6); text-align: center;">Please wait. AI is analyzing the data.</div>
    </div>

    <div class="app-container">
        
        <header class="header" data-html2canvas-ignore>
            <h1>Kaparsh</h1>
            <button id="download-btn" onclick="downloadNotes()" class="icon-btn hidden">
                <i class="fa-solid fa-arrow-down"></i>
            </button>
        </header>
            
        <!-- Tab: Home -->
        <div id="tab-home" class="tab-pane flex">
            <div class="glass-card">
                <h2 class="large-title">Chapter<br>Intelligence</h2>
                <p class="sub-title">Upload a textbook chapter for deep-extraction notes, formulas, and spaced repetition planning.</p>
            </div>

            <label id="drop-zone" for="file-upload" class="upload-zone">
                <div class="upload-icons">
                    <i class="fa-solid fa-file-pdf" style="color: var(--accent-primary);"></i>
                    <i class="fa-regular fa-image" style="color: var(--accent-secondary);"></i>
                </div>
                <div id="file-name" class="upload-text">Drag & Drop or Click to Upload</div>
                <input type="file" id="file-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
            </label>

            <button id="analyze-btn" class="btn-primary">
                Analyze Document
                <i class="fa-solid fa-wand-magic-sparkles"></i>
            </button>
        </div>

        <!-- Tab: Notes -->
        <div id="tab-summary" class="tab-pane hidden" style="background: transparent;">
            <div class="section-header">
                <span>Knowledge Graph</span>
                <span id="topic-count" class="badge">0 Concepts</span>
            </div>
            <div id="topics-grid" class="glass-card" style="padding: 32px 24px;">
                <div style="text-align: center; padding: 60px 0; opacity: 0.4;">
                    <i class="fa-solid fa-layer-group" style="font-size: 48px; margin-bottom: 20px;"></i>
                    <p style="font-size: 16px;">Your digital notebook is empty.</p>
                </div>
            </div>
        </div>

        <!-- Tab: Schedule -->
        <div id="tab-schedule" class="tab-pane hidden">
            <div id="schedule-setup">
                <div class="glass-card" style="text-align: center; padding: 40px 20px;">
                    <div style="width: 72px; height: 72px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255, 255, 255, 0.15); display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; font-size: 28px; color: #FFF;">
                        <i class="fa-solid fa-calendar-day"></i>
                    </div>
                    <h2 class="large-title">Master Planner</h2>
                    <p class="sub-title">Set your deadlines and syllabus context.</p>
                    
                    <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                        <div class="input-group">
                            <label>Target Exam Date</label>
                            <input type="date" id="schedule-exam-date">
                        </div>
                        <div class="input-group" style="margin-bottom: 24px;">
                            <label>Daily Study Hours</label>
                            <input type="number" id="schedule-study-hours" value="2" min="1" max="16" style="width: 70px;">
                        </div>

                        <label id="syllabus-drop-zone" for="syllabus-upload" class="upload-zone" style="padding: 24px;">
                            <div class="upload-icons" style="margin-bottom: 12px;"><i class="fa-solid fa-book-open"></i></div>
                            <div id="syllabus-file-name" class="upload-text" style="font-size: 14px;">Attach Syllabus Image/PDF</div>
                            <input type="file" id="syllabus-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
                        </label>

                        <div id="subject-selector-area" class="hidden" style="margin-bottom: 24px;">
                            <p style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 16px;">Detected Subjects</p>
                            <div id="subject-checkboxes" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        </div>

                        <button onclick="generateSchedule()" class="btn-primary">Generate Strategy</button>
                    </div>
                </div>
            </div>
            
            <div id="schedule-result" class="hidden flex-col">
                <div class="section-header">Study Timeline</div>
                <div class="glass-card" id="schedule-timeline-container" style="padding: 32px 24px;"></div>
            </div>
        </div>

        <!-- Tab: Quiz -->
        <div id="tab-quiz" class="tab-pane hidden">
            <div id="quiz-setup" class="glass-card" style="text-align: center; padding: 60px 20px;">
                <div style="width: 72px; height: 72px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255, 255, 255, 0.15); display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; font-size: 32px;">
                    <i class="fa-solid fa-gamepad"></i>
                </div>
                <h2 class="large-title">Knowledge Check</h2>
                <p class="sub-title">Engage active recall on processed materials.</p>
                <div style="max-width: 300px; margin: 0 auto;">
                    <button onclick="generateQuiz()" class="btn-primary">Start Assessment</button>
                </div>
            </div>
            <div id="quiz-result" class="hidden flex-col">
                <div class="section-header" style="margin-bottom: 24px;">Practice Assessment</div>
                <div id="quiz-questions-container"></div>
                <div style="max-width: 400px; margin: 30px auto;">
                    <button onclick="checkAnswers()" class="btn-primary">Submit Responses</button>
                </div>
                <div id="quiz-score-area"></div>
            </div>
        </div>

        <!-- Tab: Doubts -->
        <div id="tab-doubts" class="tab-pane chat-pane hidden">
            <div class="section-header">AI Tutor</div>
            <div id="chat-history" class="chat-history">
                <div class="chat-bubble chat-ai md-content">Hello. Ask me anything to clarify concepts from your uploaded text.</div>
            </div>
            <div class="chat-input-wrapper" data-html2canvas-ignore>
                <div class="chat-input-box">
                    <input type="text" id="doubt-input" placeholder="Type your doubt here...">
                    <button onclick="sendDoubt()" class="chat-send-btn"><i class="fa-solid fa-arrow-up"></i></button>
                </div>
            </div>
        </div>
        
    </div>

    <!-- Centralized Desktop/Mobile Nav Dock -->
    <nav class="dynamic-island" data-html2canvas-ignore>
        <div class="nav-indicator" id="nav-indicator"></div>
        <button class="nav-btn nav-active" data-target="tab-home" onclick="switchTab('tab-home', 0)"><i class="fa-solid fa-house"></i></button>
        <button class="nav-btn" data-target="tab-summary" onclick="switchTab('tab-summary', 1)"><i class="fa-solid fa-layer-group"></i></button>
        <button class="nav-btn" data-target="tab-schedule" onclick="switchTab('tab-schedule', 2)"><i class="fa-solid fa-calendar-day"></i></button>
        <button class="nav-btn" data-target="tab-quiz" onclick="switchTab('tab-quiz', 3)"><i class="fa-solid fa-gamepad"></i></button>
        <button class="nav-btn" data-target="tab-doubts" onclick="switchTab('tab-doubts', 4)"><i class="fa-solid fa-comment-dots"></i></button>
    </nav>

    <script>
        // CONFIG MARKED FOR GFM
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true
            });
        }

        // AUTO RENDER LATEX MATH HELPER
        function applyMath() {
            if (window.renderMathInElement) {
                renderMathInElement(document.body, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '\\[', right: '\\]', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false}
                    ],
                    throwOnError: false,
                    errorColor: '#EF4444'
                });
            }
        }

        const AppState = {
            extractedText: "", topics: null, schedule: null, quiz: null,
            file: null, syllabusFile: null, syllabusContextText: "", globalBannedTerms: [] 
        };

        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 14);
        document.getElementById('schedule-exam-date').valueAsDate = tomorrow;

        // Dock UI Logic
        const navBtns = document.querySelectorAll('.nav-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        const indicator = document.getElementById('nav-indicator');

        function switchTab(targetId, index) {
            if (targetId === 'tab-summary' || targetId === 'tab-quiz' || targetId === 'tab-doubts') {
                if (!AppState.topics) return alert("Please upload and analyze a document first.");
            }
            
            navBtns.forEach(btn => btn.classList.remove('nav-active'));
            navBtns[index].classList.add('nav-active');
            
            // Width of button + gap (52px + 4px = 56px step)
            indicator.style.transform = `translateX(${index * 56}px)`;

            tabPanes.forEach(pane => { pane.classList.add('hidden'); pane.classList.remove('flex'); });
            const target = document.getElementById(targetId);
            target.classList.remove('hidden');
            target.classList.add('flex');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        const toggleLoader = (show, title = 'Processing Document...', text = 'Please wait. AI is analyzing the data.') => {
            document.getElementById('loader-title').innerText = title;
            document.getElementById('loader-text').innerText = text;
            const loader = document.getElementById('global-loader');
            
            if (show) {
                loader.classList.remove('hidden');
                setTimeout(() => loader.classList.add('active'), 10);
            } else {
                loader.classList.remove('active');
                setTimeout(() => loader.classList.add('hidden'), 400);
            }
        };

        const showError = (msg) => {
            alert("Error: " + msg);
            toggleLoader(false);
            switchTab('tab-home', 0);
        };

        // File Handlers
        const fileInput = document.getElementById('file-upload');
        const fileNameDisplay = document.getElementById('file-name');
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                AppState.file = fileInput.files[0];
                fileNameDisplay.textContent = AppState.file.name;
                fileNameDisplay.style.color = '#fff';
            }
        });
        
        const syllabusInput = document.getElementById('syllabus-upload');
        const syllabusNameDisplay = document.getElementById('syllabus-file-name');
        syllabusInput.addEventListener('change', async () => {
            if (syllabusInput.files.length > 0) {
                AppState.syllabusFile = syllabusInput.files[0];
                syllabusNameDisplay.textContent = AppState.syllabusFile.name;
                await parseSyllabusSubjects();
            }
        });

        async function parseSyllabusSubjects() {
            toggleLoader(true, 'Parsing Syllabus...', 'Server is reading your syllabus & detecting subjects...');
            const formData = new FormData();
            formData.append('file', AppState.syllabusFile);

            try {
                const response = await fetch('/api/parse-syllabus', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Failed to parse syllabus");

                AppState.syllabusContextText = data.full_text;
                renderSubjectCheckboxes(data.subjects);
                toggleLoader(false);
            } catch (err) {
                toggleLoader(false);
                alert("Auto-detection notice: " + err.message + ". You can still generate the plan.");
            }
        }

        function renderSubjectCheckboxes(subjects) {
            const area = document.getElementById('subject-selector-area');
            const container = document.getElementById('subject-checkboxes');
            
            if (!subjects || subjects.length === 0) return;

            container.innerHTML = subjects.map((sub) => `
                <label class="custom-checkbox">
                    <input type="checkbox" name="syllabus-subject" value="${sub}" checked>
                    <div class="checker"><i class="fa-solid fa-check"></i></div>
                    <span class="checkbox-label">${sub}</span>
                </label>
            `).join('');
            
            area.classList.remove('hidden');
        }

        // Micro-Analyzer
        document.getElementById('analyze-btn').addEventListener('click', async () => {
            if (!AppState.file) return alert("Please upload a PDF or Image file first.");

            AppState.globalBannedTerms = []; 
            toggleLoader(true, 'Analyzing Document...', 'Establishing knowledge graph...');

            try {
                const formData = new FormData();
                formData.append('file', AppState.file);
                
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                const rawText = await response.text();
                
                if (!response.ok) {
                    let errMsg = "Network request failed.";
                    try { errMsg = JSON.parse(rawText).detail; } catch(e) {}
                    throw new Error(errMsg);
                }
                
                const data = JSON.parse(rawText);
                AppState.extractedText = data.extracted_text;
                AppState.topics = data.topics.map(t => ({ title: t.title, loaded: false }));
                
                renderTopics();
                switchTab('tab-summary', 1);
                toggleLoader(false);
                
                for (let i = 0; i < AppState.topics.length; i++) {
                    await fetchTopicDetails(i);
                    await new Promise(resolve => setTimeout(resolve, 800));
                }

                AppState.quiz = null;
                document.getElementById('quiz-result').classList.add('hidden');
                document.getElementById('quiz-result').classList.remove('flex');
                document.getElementById('quiz-setup').classList.remove('hidden');

            } catch (err) { showError(err.message); }
        });

        async function fetchTopicDetails(index) {
            const topic = AppState.topics[index];
            try {
                const response = await fetch('/api/topic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: AppState.extractedText, 
                        topic: topic.title,
                        covered_topics: AppState.globalBannedTerms.join(", ")
                    })
                });
                if (response.ok) {
                    const details = await response.json();
                    if (details.definitions && details.definitions.length > 0) {
                        details.definitions.forEach(d => AppState.globalBannedTerms.push(d.term));
                    }
                    AppState.topics[index] = { ...topic, ...details, loaded: true };
                    renderTopics(); 
                }
            } catch (e) { console.error("Failed to load topic detail", topic.title); }
        }

        function renderTopics() {
            const grid = document.getElementById('topics-grid');
            document.getElementById('topic-count').innerText = `${AppState.topics.length} Concepts`;
            
            if (AppState.topics.some(t => t.loaded)) {
                document.getElementById('download-btn').classList.remove('hidden');
            }
            
            grid.innerHTML = AppState.topics.map((t) => {
                if (!t.loaded) {
                    return `
                    <div class="note-card" style="opacity: 0.5; animation: fadeScale 1s infinite alternate;">
                        <div style="height: 24px; background: rgba(255,255,255,0.1); border-radius: 6px; width: 40%; margin-bottom: 16px;"></div>
                        <div style="height: 16px; background: rgba(255,255,255,0.05); border-radius: 6px; width: 90%; margin-bottom: 8px;"></div>
                        <div style="height: 16px; background: rgba(255,255,255,0.05); border-radius: 6px; width: 70%;"></div>
                    </div>`;
                }

                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div style="margin-top: 20px;">
                        ${t.definitions.map(d => {
                            const inlineDef = marked.parseInline(d.definition || '');
                            return `
                            <div class="def-box md-content">
                                <span class="term-title">${d.term}:</span>
                                <span>${inlineDef}</span>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const formulasHtml = (t.formulas && t.formulas.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.formulas.map(f => {
                            const mathEq = (f.equation || '').includes('$') ? f.equation : `$$${f.equation}$$`;
                            const inlineMean = marked.parseInline(f.meaning || '');
                            return `
                            <div class="formula-box">
                                <div class="eq md-content">${mathEq}</div>
                                <div class="md-content" style="font-size: 13px; color: var(--text-secondary);">${inlineMean}</div>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.derivations.map(d => {
                            const inlineTitle = marked.parseInline(d.title || '');
                            const bodyContent = marked.parse(d.content || '');
                            return `
                            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); padding: 20px; border-radius: 16px; margin-bottom: 16px;">
                                <h5 style="color: #60A5FA; font-size: 15px; margin-bottom: 12px;">${inlineTitle}</h5>
                                <div class="md-content">${bodyContent}</div>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const notesList = (t.notes || []).map(n => {
                    return `<li style="position:relative; padding-left:16px; margin-bottom:10px;">
                        <span style="position:absolute; left:0; top:8px; width:6px; height:6px; background:#fff; border-radius:50%;"></span>
                        <div class="md-content" style="display:inline;">${marked.parseInline(n)}</div>
                    </li>`;
                }).join('');

                return `
                <div class="note-card">
                    <div class="note-header">
                        <div class="note-title">${t.title}</div>
                        <div class="badge">${t.priority || 'MED'} Priority</div>
                    </div>
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    ${notesList ? `<ul style="list-style:none; padding:0; margin-top:20px;">${notesList}</ul>` : ''}
                </div>`;
            }).join('');

            // Allow elements to attach to DOM before triggering KaTeX rendering
            setTimeout(applyMath, 100);
        }

        function downloadNotes() {
            const element = document.getElementById('tab-summary');
            const opt = {
                margin: 0.5,
                filename: 'Kaparsh_Notes.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2, useCORS: true, backgroundColor: '#000' },
                jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
            };
            html2pdf().set(opt).from(element).save();
        }

        // Schedule Logic
        async function generateSchedule() {
            const examDate = document.getElementById('schedule-exam-date').value;
            const studyHours = document.getElementById('schedule-study-hours').value;
            
            if (!examDate) return alert("Please set your Target Exam Date first.");

            const selectedSubjects = [];
            document.querySelectorAll('input[name="syllabus-subject"]:checked').forEach(cb => {
                selectedSubjects.push(cb.value);
            });

            toggleLoader(true, 'Building Master Plan...', 'Mapping out timeline based on selection...');

            try {
                let response;
                if (AppState.syllabusFile) {
                    response = await fetch('/api/schedule', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            syllabus_text: AppState.syllabusContextText, 
                            exam_date: examDate, 
                            study_hours: parseFloat(studyHours),
                            selected_subjects: selectedSubjects
                        })
                    });
                } else if (AppState.topics && AppState.topics.length > 0) {
                    const loadedTopics = AppState.topics.filter(t => t.loaded);
                    response = await fetch('/api/schedule', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ topics: loadedTopics, exam_date: examDate, study_hours: parseFloat(studyHours) })
                    });
                } else {
                    throw new Error("Please upload a Syllabus OR analyze a Chapter first.");
                }

                const rawText = await response.text();
                if (!response.ok) {
                    let errMsg = "Schedule request failed.";
                    try { errMsg = JSON.parse(rawText).detail; } catch(e) {}
                    throw new Error(errMsg);
                }
                
                AppState.schedule = JSON.parse(rawText).schedule;
                renderSchedule();
                
                document.getElementById('schedule-setup').classList.add('hidden');
                document.getElementById('schedule-result').classList.remove('hidden');
                document.getElementById('schedule-result').classList.add('flex');
                toggleLoader(false);
            } catch (err) { alert("Error: " + err.message); toggleLoader(false); }
        }

        function renderSchedule() {
            const container = document.getElementById('schedule-timeline-container');
            if (!AppState.schedule || !Array.isArray(AppState.schedule)) return alert("Invalid schedule data received.");
            
            container.innerHTML = `<div style="position: relative; padding-left: 24px; margin-left: 10px; border-left: 2px solid rgba(255,255,255,0.1);">` + AppState.schedule.map((day) => `
                <div style="position: relative; margin-bottom: 32px;">
                    <div style="position: absolute; left: -39px; top: 0; width: 28px; height: 28px; background: #000; border: 2px solid #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">${day.day || '-'}</div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <span style="color: #fff; font-size: 14px; font-weight: 600;">${(day.date || '').split('-').slice(1).join('/') || 'Day'}</span>
                            <span style="color: var(--text-secondary); font-size: 13px;"><i class="fa-regular fa-clock"></i> ${day.total_hours_today || 0}h</span>
                        </div>
                        <div style="font-size: 18px; font-weight: 600; margin-bottom: 16px;">${day.focus_area || 'Study Block'}</div>
                        <div>
                            ${(day.topics || []).map(t => `
                                <div style="background: rgba(0,0,0,0.5); padding: 12px 16px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span style="font-size: 14px;">${t.name || 'Topic'}</span>
                                    <span style="font-size: 12px; color: #fff; background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 20px;">${t.estimated_minutes || 30} min</span>
                                </div>
                            `).join('')}
                        </div>
                        ${day.actionable_advice ? `<div style="font-size: 14px; color: rgba(255,255,255,0.7); margin-top: 16px; padding-left: 12px; border-left: 3px solid rgba(255,255,255,0.2);">${day.actionable_advice}</div>` : ''}
                    </div>
                </div>
            `).join('') + `</div>`;
        }

        // Quiz Logic
        async function generateQuiz() {
            toggleLoader(true, 'Generating Assessment...', 'AI is crafting complex questions based on analysis...');
            const loadedTopics = AppState.topics.filter(t => t.loaded);

            try {
                const response = await fetch('/api/quiz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topics: loadedTopics })
                });
                
                const rawText = await response.text();
                if (!response.ok) throw new Error(JSON.parse(rawText).detail || "Quiz Generation Failed");
                
                AppState.quiz = JSON.parse(rawText).quiz;
                if (!AppState.quiz || !Array.isArray(AppState.quiz)) throw new Error("AI output was invalid.");
                
                renderQuiz();
                
                document.getElementById('quiz-setup').classList.add('hidden');
                document.getElementById('quiz-result').classList.remove('hidden');
                document.getElementById('quiz-result').classList.add('flex');
                toggleLoader(false);
            } catch (err) { showError(err.message); }
        }

        function renderQuiz() {
            const container = document.getElementById('quiz-questions-container');
            document.getElementById('quiz-score-area').innerHTML = ''; 

            container.innerHTML = AppState.quiz.map((q, index) => {
                const qsText = marked.parseInline(q.question || '');
                return `
                <div id="qcard-${index}" class="glass-card q-card">
                    <div class="q-label">Question ${index + 1}</div>
                    <div class="q-text md-content">${qsText}</div>
                    <div>
                        ${(q.options || []).map(opt => `
                            <label class="quiz-opt-label">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}">
                                <div class="radio-indicator"></div>
                                <div class="md-content" style="flex:1;">${marked.parseInline(opt)}</div>
                            </label>
                        `).join('')}
                    </div>
                    <div id="result-${index}"></div>
                </div>
                `;
            }).join('');
            
            setTimeout(applyMath, 100);
        }

        function checkAnswers() {
            let score = 0;
            AppState.quiz.forEach((q, index) => {
                const selected = document.querySelector(`input[name="question-${index}"]:checked`);
                const resultDiv = document.getElementById(`result-${index}`);
                const card = document.getElementById(`qcard-${index}`);
                
                if (!selected) {
                    resultDiv.innerHTML = `<div style="color: #EF4444; font-size: 14px; font-weight: 600; margin-top: 16px;">Selection required</div>`;
                    return;
                }
                
                const explHtml = marked.parseInline(q.explanation || '');
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box correct md-content">
                            <strong style="color: #10B981; display: block; margin-bottom: 8px;">Correct</strong>
                            <span>${explHtml}</span>
                        </div>`;
                } else {
                    const corrAns = marked.parseInline(q.correct_answer || '');
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box wrong md-content">
                            <strong style="color: #EF4444; display: block; margin-bottom: 8px;">Incorrect</strong>
                            <div style="font-size: 14px; margin-bottom: 8px;">Correct Answer: <strong>${corrAns}</strong></div>
                            <span>${explHtml}</span>
                        </div>`;
                }
            });
            
            setTimeout(applyMath, 100);

            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="glass-card" style="text-align: center; padding: 40px;">
                        <p style="font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Final Assessment Score</p>
                        <div style="font-size: 56px; font-weight: 700; color: #fff; line-height: 1;">${score} <span style="font-size: 24px; color: rgba(255,255,255,0.4);">/ ${AppState.quiz.length}</span></div>
                    </div>
                `;
            }
        }

        // Chat Logic
        document.getElementById('doubt-input').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendDoubt();
        });

        async function sendDoubt() {
            const input = document.getElementById('doubt-input');
            const question = input.value.trim();
            if (!question) return;
            if (!AppState.extractedText) return alert("Please upload and process a document first.");

            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML += `<div class="chat-bubble chat-user md-content">${question}</div>`;
            input.value = '';
            
            chatHistory.scrollTop = chatHistory.scrollHeight;

            const loaderId = 'loader-' + Date.now();
            chatHistory.innerHTML += `
                <div id="${loaderId}" class="chat-bubble chat-ai" style="padding: 20px;">
                    <div class="spinner" style="width:20px; height:20px; border-width: 2px; margin: 0;"></div>
                </div>`;
            chatHistory.scrollTop = chatHistory.scrollHeight;

            try {
                const response = await fetch('/api/doubt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: AppState.extractedText, question: question })
                });
                
                const data = await response.json();
                document.getElementById(loaderId).remove();
                
                if (!response.ok) throw new Error(data.detail || "Server Error");
                
                const formattedAnswer = marked.parse(data.answer || "");
                chatHistory.innerHTML += `<div class="chat-bubble chat-ai md-content">${formattedAnswer}</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
                
                setTimeout(applyMath, 100);

            } catch (err) {
                document.getElementById(loaderId).remove();
                alert("Error: " + err.message);
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def serve_frontend():
    response = make_response(KAPARSH_FRONTEND)
    response.headers["Content-Type"] = "text/html"
    return response