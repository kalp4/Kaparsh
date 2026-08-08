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
# KAPARSH INTELLIGENCE - FULL STACK WEB UI (COLOR MORPHING AMBIENT)
# ==============================================================================

KAPARSH_FRONTEND = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#030305">
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
        /* CORE VARIABLES */
        :root {
            --bg-color: #030305;
            --surface-glass: rgba(18, 18, 24, 0.5);
            --surface-border: rgba(255, 255, 255, 0.1);
            
            --accent-primary: #8B5CF6; /* Violet */
            --accent-secondary: #0EA5E9; /* Sky */
            --accent-tertiary: #10B981; /* Emerald */
            --danger: #EF4444;
            
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.75);
            --bezier: cubic-bezier(0.16, 1, 0.3, 1);
        }

        * {
            margin: 0; padding: 0; box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
        }

        /* FIXED SCROLLING DYNAMICS: 
         * The body naturally grows, `overflow-y: auto` removes empty scrollbars, 
         * scrolling naturally occurs only when content pushes down. 
         */
        body, html {
            background-color: var(--bg-color);
            color: var(--text-primary);
            height: 100%; min-height: 100vh;
            overflow-x: hidden;
            overflow-y: auto; 
            display: flex; flex-direction: column;
            scroll-behavior: smooth;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

        /* RADIAL AMBIENT BACKGROUND WITH DYNAMIC COLOR MORPHING: 
         * Guaranteed hardware-agnostic rendering. No blurs to clip/fail. 
         * Combines dynamic drift coordinates with infinite hue-rotation.
         */
        .ambient-bg {
            position: fixed; inset: 0; z-index: 1; 
            pointer-events: none; overflow: hidden; background: #000;
        }
        .orb {
            position: absolute; border-radius: 50%; opacity: 0.6;
            mix-blend-mode: screen;
        }
        
        .orb.one { 
            width: 120vw; height: 120vw; max-width: 1200px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.45) 0%, rgba(99, 102, 241, 0) 70%); 
            top: -30%; left: -20%; 
            animation: drift-1 30s infinite alternate ease-in-out, hue-shift 20s infinite linear;
        }
        .orb.two { 
            width: 100vw; height: 100vw; max-width: 1000px;
            background: radial-gradient(circle, rgba(236, 72, 153, 0.35) 0%, rgba(236, 72, 153, 0) 70%); 
            bottom: -20%; right: -20%; 
            animation: drift-2 25s infinite alternate ease-in-out, hue-shift 35s infinite linear reverse;
        }
        .orb.three { 
            width: 90vw; height: 90vw; max-width: 900px;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.3) 0%, rgba(14, 165, 233, 0) 70%); 
            top: 20%; left: 30%; 
            animation: drift-3 35s infinite alternate ease-in-out, hue-shift 25s infinite linear;
        }

        /* Color Shifting Layer */
        @keyframes hue-shift {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        /* Unique Spatial Trajectories */
        @keyframes drift-1 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(12vw, 15vh) scale(1.1); }
            100% { transform: translate(-8vw, 10vh) scale(0.95); }
        }
        @keyframes drift-2 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-15vw, -12vh) scale(1.05); }
            100% { transform: translate(5vw, -18vh) scale(1.1); }
        }
        @keyframes drift-3 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(18vw, -15vh) scale(1.15); }
            100% { transform: translate(-10vw, 20vh) scale(0.9); }
        }

        /* Elevating content physically above background */
        .app-container {
            flex: 1 0 auto;
            width: 100%; max-width: 900px; margin: 0 auto;
            position: relative; z-index: 5;
            padding: 24px 20px calc(180px + env(safe-area-inset-bottom)) 20px;
            display: flex; flex-direction: column;
        }
        
        /* Header */
        .header {
            position: sticky; top: 0; z-index: 40;
            padding: 16px 0; margin-bottom: 24px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { 
            font-size: 26px; font-weight: 800; letter-spacing: -0.5px; 
            background: linear-gradient(90deg, #FFFFFF, #E2E8F0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .icon-btn {
            width: 44px; height: 44px; border-radius: 50%; border: none;
            background: rgba(255, 255, 255, 0.08); color: var(--text-primary);
            display: flex; align-items: center; justify-content: center; font-size: 16px;
            backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.15); 
            cursor: pointer; transition: all 0.3s var(--bezier);
        }
        .icon-btn:hover { background: rgba(255, 255, 255, 0.15); transform: translateY(-2px); }
        .icon-btn:active { transform: scale(0.9); }

        /* Typography */
        .large-title { font-size: 38px; font-weight: 700; letter-spacing: -1px; margin-bottom: 12px; line-height: 1.1; color: #fff; }
        .sub-title { font-size: 16px; color: var(--text-secondary); line-height: 1.5; font-weight: 500; margin-bottom: 28px; }
        .section-header { font-size: 22px; font-weight: 700; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between; color: #FFF; }

        .hidden { display: none !important; }
        .flex { display: flex !important; }
        .tab-pane { flex: 1; animation: fadeScale 0.4s var(--bezier); display: flex; flex-direction: column; width: 100%; }
        
        @keyframes fadeScale {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glass Modules */
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(30px) saturate(200%); -webkit-backdrop-filter: blur(30px) saturate(200%);
            border: 1px solid var(--surface-border);
            border-radius: 24px; padding: 32px;
            box-shadow: 0 16px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
            margin-bottom: 24px; position: relative;
        }

        /* Colorful Upload Zones */
        .upload-zone {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 48px 24px; border-radius: 20px;
            background: rgba(255, 255, 255, 0.04);
            border: 2px dashed rgba(255, 255, 255, 0.25);
            cursor: pointer; transition: all 0.3s var(--bezier);
            text-align: center; margin-bottom: 28px; width: 100%;
        }
        .upload-zone:hover { 
            background: rgba(139, 92, 246, 0.1); border-color: #A78BFA; 
            box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2); transform: scale(0.99);
        }
        .upload-zone input[type="file"] { position: absolute; opacity: 0; width: 0; height: 0; }
        .upload-icons { display: flex; gap: 20px; margin-bottom: 16px; }
        .upload-icons i { font-size: 38px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4)); }
        .upload-text { font-size: 17px; font-weight: 600; color: #FFF; }

        /* Radiant Primary Buttons */
        .btn-primary {
            width: 100%; padding: 20px; border-radius: 18px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(219, 39, 119, 0.8));
            color: #fff; text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.3); font-size: 17px; font-weight: 700; letter-spacing: 0.5px;
            cursor: pointer; transition: all 0.3s var(--bezier);
            display: flex; justify-content: center; align-items: center; gap: 12px;
            box-shadow: 0 8px 30px rgba(219, 39, 119, 0.3); backdrop-filter: blur(10px);
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, rgba(99, 102, 241, 1), rgba(219, 39, 119, 1));
            border-color: #fff; transform: translateY(-3px); box-shadow: 0 12px 40px rgba(219, 39, 119, 0.5);
        }
        .btn-primary:active { transform: scale(0.96); box-shadow: 0 4px 15px rgba(219, 39, 119, 0.3); }

        /* Input Elements */
        .input-group {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px; padding: 18px 24px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
        }
        .input-group label { font-size: 16px; color: #fff; font-weight: 600; }
        .input-group input { 
            background: transparent; border: none; outline: none; color: var(--accent-secondary); 
            font-size: 18px; font-weight: 700; text-align: right; 
            -webkit-appearance: none; appearance: none;
        }

        .custom-checkbox {
            display: flex; align-items: center; gap: 16px; padding: 18px 20px;
            background: rgba(255,255,255,0.05); border: 1px solid var(--surface-border);
            border-radius: 16px; transition: 0.2s var(--bezier); cursor: pointer; margin-bottom: 12px;
        }
        .custom-checkbox:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.25); }
        .custom-checkbox input { display: none; }
        .checker {
            width: 26px; height: 26px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.4);
            display: flex; align-items: center; justify-content: center; transition: 0.3s;
        }
        .checker i { font-size: 13px; color: #fff; opacity: 0; transform: scale(0.5); transition: 0.3s; }
        .custom-checkbox input:checked + .checker { background: var(--accent-primary); border-color: var(--accent-primary); box-shadow: 0 0 16px rgba(139, 92, 246, 0.6); }
        .custom-checkbox input:checked + .checker i { opacity: 1; transform: scale(1); }
        .checkbox-label { font-size: 16px; font-weight: 600; flex: 1; color: #fff; }

        /* Rich Knowledge Graph / MarkDown System */
        .md-content { line-height: 1.7; color: #F1F5F9; word-wrap: break-word; overflow-wrap: anywhere; }
        .md-content p { margin-bottom: 16px; }
        .md-content p:last-child { margin-bottom: 0; }
        .md-content strong { color: #fff; font-weight: 700; }
        .md-content em { color: #CBD5E1; }
        .md-content ul, .md-content ol { padding-left: 20px; margin-bottom: 16px; }
        .md-content li { margin-bottom: 8px; }
        .md-content code { background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 8px; font-family: ui-monospace, monospace; font-size: 0.9em; color: #C4B5FD; }
        .md-content pre { background: rgba(0,0,0,0.65); padding: 20px; border-radius: 16px; overflow-x: auto; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.1); }
        .md-content pre code { background: transparent; padding: 0; border: none; font-size: 14px; color: #E2E8F0; }
        .md-content h3, .md-content h4 { color: #fff; font-weight: 700; margin-top: 24px; margin-bottom: 14px; }
        
        .katex { color: #FFF; font-size: 1.05em; } 
        .katex-display { margin: 18px 0; overflow-x: auto; overflow-y: hidden; background: rgba(0,0,0,0.4); padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); text-align: center; }

        /* Grid Layouts for Desktop Quiz */
        @media (min-width: 768px) {
            #quiz-questions-container { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
        }

        /* Vivid Notes Display */
        .badge {
            font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
            padding: 8px 16px; border-radius: 20px;
            background: linear-gradient(90deg, var(--accent-primary), #DB2777); 
            color: #FFF; box-shadow: 0 4px 12px rgba(219, 39, 119, 0.4);
        }
        .note-card { border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 32px; margin-bottom: 32px; }
        .note-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .note-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }
        .note-title { font-size: 22px; font-weight: 700; color: #fff; line-height: 1.3; }
        
        .def-box {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.15), transparent); 
            border-left: 4px solid var(--accent-primary);
            padding: 16px 20px; border-radius: 0 16px 16px 0; margin-bottom: 16px;
        }
        .def-box span.term-title { font-weight: 700; color: #C4B5FD; margin-right: 8px; font-size: 16px; }

        .formula-box {
            background: rgba(0,0,0,0.6); border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 20px; border-radius: 16px; text-align: center; margin-bottom: 16px;
            box-shadow: inset 0 2px 16px rgba(0,0,0,0.5);
        }
        .formula-box .eq { font-size: 20px; font-weight: bold; color: #34D399; margin-bottom: 8px; }

        /* ASSESSEMENT/QUIZ UI FIXES - 100% RELIABLE GRID & WRAPPING */
        .q-card { 
            border: 1px solid rgba(255,255,255,0.15); margin-bottom: 24px; padding: 28px; 
            display: flex; flex-direction: column; height: 100%;
        }
        @media (min-width: 768px) { .q-card { margin-bottom: 0 !important; } }
        
        .q-label { font-size: 14px; font-weight: 800; color: #38BDF8; letter-spacing: 1px; margin-bottom: 14px; text-transform: uppercase; }
        .q-text.md-content { font-size: 18px; font-weight: 600; margin-bottom: 24px; line-height: 1.4; color: #FFF; }
        
        .quiz-opt-label {
            display: flex; align-items: flex-start; gap: 16px; padding: 16px 20px; margin-bottom: 14px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px; font-size: 16px; cursor: pointer; transition: all 0.2s var(--bezier);
            width: 100%;
        }
        .quiz-opt-label .md-content { flex: 1; min-width: 0; }
        .quiz-opt-label .md-content p { margin: 0 !important; } 
        
        .quiz-opt-label input { display: none; }
        .radio-indicator { width: 24px; height: 24px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.4); flex-shrink: 0; margin-top: 2px; transition: 0.2s; }
        
        .quiz-opt-label:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.3); }
        .quiz-opt-label:has(input:checked) { background: rgba(56, 189, 248, 0.15); border-color: #38BDF8; }
        .quiz-opt-label:has(input:checked) .radio-indicator { border-color: #38BDF8; border-width: 6px; background: #fff; }
        
        .quiz-res-box { padding: 18px; border-radius: 16px; font-size: 16px; font-weight: 500; margin-top: auto; }
        .quiz-res-box.correct { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); }
        .quiz-res-box.wrong { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); }
        /* Adding visual gap between options block and result block securely */
        .q-card-opts { margin-bottom: 24px; }

        /* Chat Optimization */
        .chat-pane { display: flex; flex-direction: column; overflow: hidden; padding-bottom: 60px; min-height: 60vh; }
        .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; padding-bottom: 30px; }
        .chat-bubble { max-width: 82%; padding: 18px 22px; border-radius: 20px; font-size: 16px; animation: fadeScale 0.3s var(--bezier); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        
        .chat-ai {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
            border: 1px solid rgba(255,255,255,0.15);
            align-self: flex-start; border-bottom-left-radius: 6px; 
            backdrop-filter: blur(24px);
        }
        .chat-user { 
            background: linear-gradient(135deg, var(--accent-primary), #DB2777); border: 1px solid rgba(255,255,255,0.3);
            align-self: flex-end; border-bottom-right-radius: 6px; color: #fff; font-weight: 500;
        }

        /* 
         * HUGE, PREMIUM FLOATING DOCK (Dynamic Island)
         */
        .dynamic-island {
            position: fixed; 
            bottom: calc(24px + env(safe-area-inset-bottom)); 
            left: 50%; transform: translateX(-50%);
            background: rgba(10, 10, 14, 0.8); 
            backdrop-filter: blur(50px) saturate(250%); -webkit-backdrop-filter: blur(50px) saturate(250%);
            border: 1px solid rgba(255, 255, 255, 0.2); 
            border-radius: 100px;
            display: flex; 
            padding: 12px; gap: 12px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.8), inset 0 2px 4px rgba(255,255,255,0.1);
            z-index: 100; max-width: max-content;
        }
        .nav-indicator {
            position: absolute; top: 12px; left: 12px; 
            width: 64px; height: 64px; /* MASSIVE */
            background: rgba(255,255,255,0.25); border-radius: 50%;
            transition: transform 0.4s var(--bezier); z-index: 0; pointer-events: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .nav-btn {
            width: 64px; height: 64px; 
            border-radius: 50%; border: none; background: transparent;
            color: rgba(255,255,255,0.5); display: flex; align-items: center; justify-content: center;
            font-size: 24px; cursor: pointer; position: relative; z-index: 1;
            transition: all 0.3s;
        }
        .nav-btn:hover { color: #fff; }
        .nav-btn.nav-active { color: #fff; transform: scale(1.08); text-shadow: 0 0 16px rgba(255,255,255,0.6); }

        /* Chat Floating Bar */
        .chat-input-wrapper {
            position: sticky; bottom: 0px; 
            width: 100%; z-index: 40; padding-bottom: 24px;
            background: linear-gradient(0deg, var(--bg-color) 40%, transparent);
        }
        .chat-input-box {
            background: rgba(15, 15, 20, 0.9); backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
            border: 1px solid rgba(255,255,255,0.25); border-radius: 40px;
            display: flex; align-items: center; padding: 10px 10px 10px 24px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.7);
        }
        .chat-input-box input {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 17px; font-weight: 500; width: 100%;
        }
        .chat-send-btn {
            width: 48px; height: 48px; border-radius: 50%; 
            background: linear-gradient(135deg, #8B5CF6, #EC4899);
            border: none; color: #FFF; display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: 0.2s; font-size: 18px; margin-left: 12px;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
        }
        .chat-send-btn:hover { transform: scale(1.05); filter: brightness(1.1); }
        .chat-send-btn:active { transform: scale(0.9); }

        /* Ultra Premium Loader */
        .loader-overlay {
            position: fixed; inset: 0; z-index: 9999;
            background: rgba(3, 3, 5, 0.85); backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            opacity: 0; pointer-events: none; transition: opacity 0.4s ease;
        }
        .loader-overlay.active { opacity: 1; pointer-events: auto; }
        .spinner {
            width: 64px; height: 64px; border-radius: 50%;
            border: 5px solid rgba(255,255,255,0.15);
            border-top-color: var(--accent-secondary); animation: spin 0.8s infinite cubic-bezier(0.5, 0, 0.5, 1);
            margin-bottom: 28px;
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
        <div id="loader-title" style="font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 12px; letter-spacing: 0.5px;">Processing Data...</div>
        <div id="loader-text" style="font-size: 16px; color: rgba(255,255,255,0.8); text-align: center; max-width: 340px; line-height: 1.5;">Please wait. Artificial intelligence is establishing a knowledge graph.</div>
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
            <div class="glass-card" style="border-left: 5px solid var(--accent-primary);">
                <h2 class="large-title">Document<br>Intelligence</h2>
                <p class="sub-title">Upload a textbook chapter for deep extraction, complex formulas, and high-level knowledge synthesis.</p>
            </div>

            <label id="drop-zone" for="file-upload" class="upload-zone">
                <div class="upload-icons">
                    <i class="fa-solid fa-file-pdf" style="color: #A78BFA;"></i>
                    <i class="fa-regular fa-image" style="color: #38BDF8;"></i>
                </div>
                <div id="file-name" class="upload-text">Drag & Drop or Tap to Upload</div>
                <input type="file" id="file-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
            </label>

            <button id="analyze-btn" class="btn-primary">
                Synthesize Knowledge
                <i class="fa-solid fa-wand-magic-sparkles"></i>
            </button>
        </div>

        <!-- Tab: Notes -->
        <div id="tab-summary" class="tab-pane hidden" style="background: transparent;">
            <div class="section-header">
                <span>Knowledge Graph</span>
                <span id="topic-count" class="badge">0 Concepts</span>
            </div>
            <div id="topics-grid" class="glass-card" style="padding: 40px 32px;">
                <div style="text-align: center; padding: 80px 0; opacity: 0.5;">
                    <i class="fa-solid fa-layer-group" style="font-size: 64px; margin-bottom: 24px;"></i>
                    <p style="font-size: 18px; font-weight: 500;">Your structured notebook is empty.</p>
                </div>
            </div>
        </div>

        <!-- Tab: Schedule -->
        <div id="tab-schedule" class="tab-pane hidden">
            <div id="schedule-setup">
                <div class="glass-card" style="text-align: center; padding: 48px 24px;">
                    <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(219, 39, 119, 0.2)); border: 2px solid rgba(239, 68, 68, 0.4); display: flex; align-items: center; justify-content: center; margin: 0 auto 24px; font-size: 34px; color: #FCA5A5; box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);">
                        <i class="fa-solid fa-calendar-day"></i>
                    </div>
                    <h2 class="large-title">Master Planner</h2>
                    <p class="sub-title">Set deadlines and inject a syllabus for strategy.</p>
                    
                    <div style="text-align: left; max-width: 440px; margin: 0 auto;">
                        <div class="input-group">
                            <label>Target Exam Date</label>
                            <input type="date" id="schedule-exam-date">
                        </div>
                        <div class="input-group" style="margin-bottom: 28px;">
                            <label>Daily Study Hours</label>
                            <input type="number" id="schedule-study-hours" value="2" min="1" max="16" style="width: 80px;">
                        </div>

                        <label id="syllabus-drop-zone" for="syllabus-upload" class="upload-zone" style="padding: 28px;">
                            <div class="upload-icons" style="margin-bottom: 16px;"><i class="fa-solid fa-book-open" style="color: #6EE7B7;"></i></div>
                            <div id="syllabus-file-name" class="upload-text" style="font-size: 15px;">Attach Syllabus Material</div>
                            <input type="file" id="syllabus-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
                        </label>

                        <div id="subject-selector-area" class="hidden" style="margin-bottom: 28px;">
                            <p style="font-size: 14px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 16px; letter-spacing: 1px;">Detected Subjects</p>
                            <div id="subject-checkboxes" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        </div>

                        <button onclick="generateSchedule()" class="btn-primary">Generate Strategy</button>
                    </div>
                </div>
            </div>
            
            <div id="schedule-result" class="hidden flex-col">
                <div class="section-header">Action Timeline</div>
                <div class="glass-card" id="schedule-timeline-container" style="padding: 32px 28px;"></div>
            </div>
        </div>

        <!-- Tab: Quiz -->
        <div id="tab-quiz" class="tab-pane hidden">
            <div id="quiz-setup" class="glass-card" style="text-align: center; padding: 70px 24px;">
                <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(56, 189, 248, 0.2)); border: 2px solid rgba(56, 189, 248, 0.4); display: flex; align-items: center; justify-content: center; margin: 0 auto 24px; font-size: 34px; color: #7DD3FC; box-shadow: 0 10px 30px rgba(56, 189, 248, 0.2);">
                    <i class="fa-solid fa-gamepad"></i>
                </div>
                <h2 class="large-title">Active Recall</h2>
                <p class="sub-title">Test your mastery of the analyzed chapter intelligently.</p>
                <div style="max-width: 340px; margin: 0 auto;">
                    <button onclick="generateQuiz()" class="btn-primary" style="background: linear-gradient(135deg, #0EA5E9, #10B981);">Start Assessment</button>
                </div>
            </div>
            <div id="quiz-result" class="hidden flex-col">
                <div class="section-header" style="margin-bottom: 24px;">Practice Assessment</div>
                <div id="quiz-questions-container"></div>
                <div style="max-width: 440px; margin: 40px auto 0;">
                    <button onclick="checkAnswers()" class="btn-primary" style="background: linear-gradient(135deg, #38BDF8, #8B5CF6);">Submit Responses</button>
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

    <!-- Massive Centered Dock Navigation -->
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

        // Dock UI Logic - Massive Indicator calculations
        const navBtns = document.querySelectorAll('.nav-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        const indicator = document.getElementById('nav-indicator');

        function switchTab(targetId, index) {
            if (targetId === 'tab-summary' || targetId === 'tab-quiz' || targetId === 'tab-doubts') {
                if (!AppState.topics) return alert("Please upload and analyze a document first.");
            }
            
            navBtns.forEach(btn => btn.classList.remove('nav-active'));
            navBtns[index].classList.add('nav-active');
            
            // Width of enlarged button (64px) + Gap (12px) = 76px transformation step
            indicator.style.transform = `translateX(${index * 76}px)`;

            tabPanes.forEach(pane => { pane.classList.add('hidden'); pane.classList.remove('flex'); });
            const target = document.getElementById(targetId);
            target.classList.remove('hidden');
            target.classList.add('flex');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        const toggleLoader = (show, title = 'Processing Document...', text = 'Please wait. AI is actively analyzing the data.') => {
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
            toggleLoader(true, 'Synthesizing Knowledge...', 'Establishing graph and parsing content...');

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
                        <div style="height: 28px; background: rgba(255,255,255,0.15); border-radius: 8px; width: 40%; margin-bottom: 20px;"></div>
                        <div style="height: 18px; background: rgba(255,255,255,0.08); border-radius: 6px; width: 90%; margin-bottom: 12px;"></div>
                        <div style="height: 18px; background: rgba(255,255,255,0.08); border-radius: 6px; width: 70%;"></div>
                    </div>`;
                }

                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div style="margin-top: 24px;">
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
                    <div style="margin-top: 20px;">
                        ${t.formulas.map(f => {
                            const mathEq = (f.equation || '').includes('$') ? f.equation : `$$${f.equation}$$`;
                            const inlineMean = marked.parseInline(f.meaning || '');
                            return `
                            <div class="formula-box">
                                <div class="eq md-content">${mathEq}</div>
                                <div class="md-content" style="font-size: 14px; color: #CBD5E1; font-weight: 500;">${inlineMean}</div>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div style="margin-top: 20px;">
                        ${t.derivations.map(d => {
                            const inlineTitle = marked.parseInline(d.title || '');
                            const bodyContent = marked.parse(d.content || '');
                            return `
                            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); padding: 24px; border-radius: 20px; margin-bottom: 20px;">
                                <h5 style="color: #38BDF8; font-size: 16px; margin-bottom: 16px;">${inlineTitle}</h5>
                                <div class="md-content">${bodyContent}</div>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const notesList = (t.notes || []).map(n => {
                    return `<li style="position:relative; padding-left:20px; margin-bottom:12px;">
                        <span style="position:absolute; left:0; top:10px; width:8px; height:8px; background:#fff; border-radius:50%;"></span>
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
                    ${notesList ? `<ul style="list-style:none; padding:0; margin-top:24px;">${notesList}</ul>` : ''}
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
                html2canvas: { scale: 2, useCORS: true, backgroundColor: '#030305' },
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
            
            container.innerHTML = `<div style="position: relative; padding-left: 28px; margin-left: 10px; border-left: 2px solid rgba(255,255,255,0.15);">` + AppState.schedule.map((day) => `
                <div style="position: relative; margin-bottom: 36px;">
                    <div style="position: absolute; left: -43px; top: 0; width: 32px; height: 32px; background: #000; border: 3px solid var(--accent-primary); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800;">${day.day || '-'}</div>
                    <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 24px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">
                            <span style="color: #6EE7B7; font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">${(day.date || '').split('-').slice(1).join('/') || 'Day'}</span>
                            <span style="color: var(--text-secondary); font-size: 15px; font-weight: 600;"><i class="fa-regular fa-clock"></i> ${day.total_hours_today || 0}h</span>
                        </div>
                        <div style="font-size: 20px; font-weight: 700; margin-bottom: 18px; color: #FFF;">${day.focus_area || 'Study Block'}</div>
                        <div>
                            ${(day.topics || []).map(t => `
                                <div style="background: rgba(0,0,0,0.5); padding: 14px 18px; border-radius: 14px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                                    <span style="font-size: 15px; font-weight: 500;">${t.name || 'Topic'}</span>
                                    <span style="font-size: 13px; color: #fff; background: linear-gradient(135deg, rgba(124, 58, 237, 0.5), rgba(219, 39, 119, 0.5)); padding: 6px 12px; border-radius: 20px; font-weight: 700;">${t.estimated_minutes || 30} min</span>
                                </div>
                            `).join('')}
                        </div>
                        ${day.actionable_advice ? `<div style="font-size: 15px; color: rgba(255,255,255,0.8); margin-top: 18px; padding-left: 14px; border-left: 3px solid rgba(56, 189, 248, 0.8); line-height: 1.5;">${day.actionable_advice}</div>` : ''}
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
                const qsText = marked.parse(q.question || '');
                return `
                <div id="qcard-${index}" class="glass-card q-card">
                    <div class="q-label">Question ${index + 1}</div>
                    <div class="q-text md-content">${qsText}</div>
                    <div class="q-card-opts">
                        ${(q.options || []).map(opt => `
                            <label class="quiz-opt-label">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}">
                                <div class="radio-indicator"></div>
                                <div class="md-content">${marked.parse(opt)}</div>
                            </label>
                        `).join('')}
                    </div>
                    <div id="result-${index}" style="margin-top: auto; display:none;"></div>
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
                
                if (!selected) {
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<div style="color: #EF4444; font-size: 15px; font-weight: 700; margin-top: 10px;">Selection required</div>`;
                    return;
                }
                
                const explHtml = marked.parseInline(q.explanation || '');
                resultDiv.style.display = 'block';
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box correct md-content">
                            <strong style="color: #10B981; display: block; margin-bottom: 8px; font-size: 17px;">Correct</strong>
                            <span>${explHtml}</span>
                        </div>`;
                } else {
                    const corrAns = marked.parseInline(q.correct_answer || '');
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box wrong md-content">
                            <strong style="color: #EF4444; display: block; margin-bottom: 8px; font-size: 17px;">Incorrect</strong>
                            <div style="font-size: 15px; margin-bottom: 10px;">Correct Answer: <strong>${corrAns}</strong></div>
                            <span>${explHtml}</span>
                        </div>`;
                }
            });
            
            setTimeout(applyMath, 100);

            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="glass-card" style="text-align: center; padding: 48px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.2);">
                        <p style="font-size: 16px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 16px;">Final Assessment Score</p>
                        <div style="font-size: 64px; font-weight: 800; color: #fff; line-height: 1;">${score} <span style="font-size: 30px; color: rgba(255,255,255,0.4);">/ ${AppState.quiz.length}</span></div>
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
                <div id="${loaderId}" class="chat-bubble chat-ai" style="padding: 24px;">
                    <div class="spinner" style="width:24px; height:24px; border-width: 3px; margin: 0; border-top-color: var(--accent-primary);"></div>
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