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
# MOBILE NATIVE UI INJECTION
# ==============================================================================

KAPARSH_FRONTEND = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#000000">
    <title>Kaparsh</title>
    <!-- Keeping external scripts for critical functionality only -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <style>
        /* APPLE VISION-OS / NATIVE iOS DESIGN SYSTEM */
        
        :root {
            --bg-color: #000000;
            --surface-color: rgba(28, 28, 30, 0.6);
            --surface-border: rgba(255, 255, 255, 0.08);
            --accent: #5865F2;
            --accent-glow: rgba(88, 101, 242, 0.4);
            --success: #32D74B;
            --danger: #FF453A;
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.6);
            --text-tertiary: rgba(255, 255, 255, 0.4);
            --bezier: cubic-bezier(0.2, 0.8, 0.2, 1);
        }

        * {
            margin: 0; padding: 0; box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", system-ui, sans-serif;
        }

        body, html {
            background-color: var(--bg-color);
            color: var(--text-primary);
            overflow-x: hidden;
            width: 100%; height: 100%;
        }

        /* Ambient Liquid Background */
        .ambient-bg {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: -1; overflow: hidden; pointer-events: none;
        }
        .orb {
            position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.15;
            animation: float 20s infinite alternate ease-in-out;
        }
        .orb.blue { width: 60vw; height: 60vw; background: var(--accent); top: -10%; left: -10%; }
        .orb.green { width: 50vw; height: 50vw; background: var(--success); bottom: -10%; right: -10%; animation-delay: -10s; }
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(15vw, 15vh) scale(1.2); }
        }

        /* Core Layout */
        .app-container {
            max-width: 500px; margin: 0 auto; min-height: 100vh;
            position: relative; padding-bottom: 120px;
        }
        
        .header {
            position: sticky; top: 0; z-index: 40; padding: 16px 24px;
            display: flex; justify-content: space-between; align-items: center;
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
            border-bottom: 0.5px solid var(--surface-border);
        }
        .header h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }

        .icon-btn {
            width: 36px; height: 36px; border-radius: 50%; border: none;
            background: var(--surface-color); color: var(--text-primary);
            display: flex; align-items: center; justify-content: center;
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 0.5px solid var(--surface-border); cursor: pointer;
            transition: transform 0.2s var(--bezier);
        }
        .icon-btn:active { transform: scale(0.9); }

        /* Typography Hierarchy */
        .large-title { font-size: 32px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; line-height: 1.1; }
        .sub-title { font-size: 15px; color: var(--text-secondary); line-height: 1.4; font-weight: 400; margin-bottom: 24px; }
        .section-header { font-size: 20px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }

        /* Utilities */
        .hidden { display: none !important; }
        .flex { display: flex !important; }
        .tab-pane { padding: 24px; animation: fadeUp 0.6s var(--bezier); }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glass Cards */
        .glass-card {
            background: var(--surface-color);
            backdrop-filter: blur(20px) saturate(150%); -webkit-backdrop-filter: blur(20px) saturate(150%);
            border: 0.5px solid var(--surface-border);
            border-radius: 28px; padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            transition: transform 0.3s var(--bezier), box-shadow 0.3s var(--bezier);
            position: relative; overflow: hidden;
        }

        /* Upload Zones */
        .upload-zone {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px 20px; border-radius: 28px;
            background: rgba(255, 255, 255, 0.03);
            border: 1.5px dashed rgba(255, 255, 255, 0.15);
            cursor: pointer; transition: all 0.3s var(--bezier);
            position: relative; margin-bottom: 24px; text-align: center;
        }
        .upload-zone:active { transform: scale(0.98); background: rgba(255, 255, 255, 0.06); }
        .upload-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; z-index: 10; }
        .upload-icons { display: flex; gap: 16px; margin-bottom: 16px; }
        .upload-icons i { font-size: 28px; }
        .upload-icons .fa-file-pdf { color: var(--accent); }
        .upload-icons .fa-image { color: var(--success); }
        .upload-text { font-size: 16px; font-weight: 600; color: var(--text-primary); }

        /* Primary Button */
        .btn-primary {
            width: 100%; padding: 18px; border-radius: 100px;
            background: linear-gradient(135deg, var(--accent), #4752C4);
            border: none; color: #fff; font-size: 17px; font-weight: 600;
            cursor: pointer; transition: all 0.3s var(--bezier);
            box-shadow: 0 10px 20px var(--accent-glow);
            display: flex; justify-content: center; align-items: center; gap: 8px;
        }
        .btn-primary:active { transform: scale(0.96); box-shadow: 0 5px 10px var(--accent-glow); }
        .btn-primary.success-style { background: linear-gradient(135deg, var(--success), #28B43C); color: #000; box-shadow: 0 10px 20px rgba(50, 215, 75, 0.2); }

        /* Native Inputs */
        .input-group {
            background: rgba(0,0,0,0.3); border: 0.5px solid var(--surface-border);
            border-radius: 16px; padding: 14px 20px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
        }
        .input-group label { font-size: 15px; color: var(--text-secondary); font-weight: 500; }
        .input-group input { 
            background: transparent; border: none; outline: none; color: #fff; 
            font-size: 16px; font-weight: 600; text-align: right; font-family: inherit;
        }
        .input-group input[type="date"] { color: var(--success); color-scheme: dark; }

        /* Checkboxes (Syllabus) */
        .checkbox-list { display: flex; flex-direction: column; gap: 12px; max-height: 250px; overflow-y: auto; margin-bottom: 24px; padding-right: 8px; }
        .custom-checkbox {
            display: flex; align-items: center; gap: 16px; padding: 14px 16px;
            background: rgba(255,255,255,0.03); border: 0.5px solid var(--surface-border);
            border-radius: 16px; cursor: pointer; transition: 0.2s var(--bezier);
        }
        .custom-checkbox input { display: none; }
        .checker {
            width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid var(--text-tertiary);
            display: flex; align-items: center; justify-content: center; transition: 0.2s var(--bezier);
        }
        .checker i { font-size: 10px; color: #000; opacity: 0; transform: scale(0.5); transition: 0.2s var(--bezier); }
        .custom-checkbox input:checked + .checker { background: var(--accent); border-color: var(--accent); }
        .custom-checkbox input:checked + .checker i { opacity: 1; transform: scale(1); }
        .checkbox-label { font-size: 15px; font-weight: 500; flex: 1; }

        /* Notes UI */
        .badge {
            font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
            padding: 4px 10px; border-radius: 100px;
            background: rgba(255, 255, 255, 0.1); color: var(--text-secondary);
        }
        .note-card {
            border-bottom: 0.5px solid var(--surface-border); padding-bottom: 24px; margin-bottom: 24px;
        }
        .note-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .note-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
        .note-title { font-size: 19px; font-weight: 600; color: #fff; line-height: 1.3; max-width: 80%; }
        
        .def-box {
            background: rgba(88, 101, 242, 0.1); border-left: 3px solid var(--accent);
            padding: 12px 16px; border-radius: 0 12px 12px 0; margin-bottom: 12px;
        }
        .def-box span { font-weight: 600; color: var(--accent); margin-right: 8px; }
        .def-box p { font-size: 14px; color: #fff; display: inline; }

        .formula-box {
            background: rgba(0,0,0,0.4); border: 0.5px solid var(--surface-border);
            padding: 16px; border-radius: 16px; text-align: center; margin-bottom: 12px;
        }
        .formula-box .eq { font-family: ui-monospace, monospace; font-size: 18px; font-weight: 700; color: var(--success); margin-bottom: 4px; }
        .formula-box .meaning { font-size: 12px; color: var(--text-secondary); }

        .deriv-box {
            background: rgba(255,255,255,0.02); border: 0.5px solid var(--surface-border);
            padding: 16px; border-radius: 16px; margin-bottom: 12px;
        }
        .deriv-box h5 { font-size: 13px; color: var(--accent); margin-bottom: 8px; font-weight: 600; }
        .deriv-box pre { font-family: ui-monospace, monospace; font-size: 13px; color: var(--text-secondary); white-space: pre-wrap; }

        .note-list { list-style: none; padding-left: 4px; }
        .note-list li {
            position: relative; padding-left: 20px; font-size: 14px; color: var(--text-primary);
            line-height: 1.5; margin-bottom: 8px;
        }
        .note-list li::before {
            content: ''; position: absolute; left: 0; top: 7px; width: 6px; height: 6px;
            border-radius: 50%; background: var(--success);
        }

        /* Timeline (Schedule) */
        .timeline { position: relative; padding-left: 28px; margin-top: 16px; }
        .timeline::before {
            content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 1.5px;
            background: linear-gradient(to bottom, var(--accent) 0%, rgba(255,255,255,0.05) 100%);
        }
        .timeline-item { position: relative; margin-bottom: 32px; }
        .timeline-dot {
            position: absolute; left: -28px; top: 12px; width: 22px; height: 22px;
            border-radius: 50%; background: var(--bg-color); border: 2px solid var(--accent);
            display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700;
        }
        .timeline-content {
            background: var(--surface-color); border: 0.5px solid var(--surface-border);
            border-radius: 20px; padding: 16px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        }
        .tl-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .tl-date { color: var(--success); font-size: 13px; font-weight: 700; }
        .tl-hours { color: var(--text-secondary); font-size: 11px; font-weight: 600; display: flex; align-items: center; gap: 4px; }
        .tl-focus { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
        .tl-topic {
            background: rgba(0,0,0,0.5); padding: 10px 12px; border-radius: 12px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
        }
        .tl-topic-name { font-size: 13px; font-weight: 500; }
        .tl-topic-time { font-size: 11px; color: var(--accent); font-weight: 700; background: rgba(88, 101, 242, 0.15); padding: 4px 8px; border-radius: 6px; }
        .tl-advice { font-size: 13px; color: var(--text-secondary); margin-top: 12px; padding-left: 10px; border-left: 2px solid var(--surface-border); }

        /* Quiz */
        .q-card { margin-bottom: 24px; transition: 0.3s var(--bezier); }
        .q-label { font-size: 12px; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; margin-bottom: 8px; }
        .q-text { font-size: 18px; font-weight: 600; line-height: 1.4; margin-bottom: 20px; }
        
        .quiz-opt-label {
            display: block; padding: 16px 20px; margin-bottom: 12px;
            background: rgba(255,255,255,0.03); border: 1px solid var(--surface-border);
            border-radius: 16px; font-size: 15px; font-weight: 500;
            cursor: pointer; transition: all 0.2s var(--bezier); position: relative; overflow: hidden;
        }
        .quiz-opt-label input { display: none; }
        .quiz-opt-label:active { transform: scale(0.98); }
        .quiz-opt-label:has(input:checked) {
            background: rgba(88, 101, 242, 0.15); border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent);
        }
        .quiz-res-box { padding: 16px; border-radius: 16px; margin-top: 16px; font-size: 14px; line-height: 1.5; }
        .quiz-res-box.correct { background: rgba(50, 215, 75, 0.1); border: 1px solid rgba(50, 215, 75, 0.3); }
        .quiz-res-box.wrong { background: rgba(255, 69, 58, 0.1); border: 1px solid rgba(255, 69, 58, 0.3); }

        /* Chat */
        .chat-container { display: flex; flex-direction: column; height: calc(100vh - 180px); }
        .chat-history { flex: 1; overflow-y: auto; padding-bottom: 20px; display: flex; flex-direction: column; gap: 12px; scroll-behavior: smooth; }
        .chat-history::-webkit-scrollbar { display: none; }
        
        .chat-bubble { max-width: 85%; padding: 12px 16px; border-radius: 20px; font-size: 15px; line-height: 1.4; animation: bubbleIn 0.3s var(--bezier); }
        @keyframes bubbleIn { from { opacity: 0; transform: scale(0.9) translateY(10px); } to { opacity: 1; transform: scale(1) translateY(0); } }
        
        .chat-ai {
            background: var(--surface-color); border: 0.5px solid var(--surface-border);
            align-self: flex-start; border-bottom-left-radius: 4px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        }
        .chat-user {
            background: var(--accent); color: #fff;
            align-self: flex-end; border-bottom-right-radius: 4px;
        }
        .typing-dots { display: flex; gap: 4px; padding: 4px; }
        .typing-dots span { width: 6px; height: 6px; background: var(--text-secondary); border-radius: 50%; animation: typeDot 1.4s infinite ease-in-out both; }
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typeDot { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

        .chat-input-wrapper {
            position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%);
            width: 100%; max-width: 460px; padding: 0 16px; z-index: 40;
        }
        .chat-input-box {
            background: rgba(30, 30, 30, 0.85); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
            border: 0.5px solid var(--surface-border); border-radius: 100px;
            display: flex; align-items: center; padding: 6px 6px 6px 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .chat-input-box input {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 15px; font-family: inherit;
        }
        .chat-input-box input::placeholder { color: var(--text-tertiary); }
        .chat-send-btn {
            width: 36px; height: 36px; border-radius: 50%; background: var(--accent);
            border: none; color: #fff; display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: 0.2s var(--bezier);
        }
        .chat-send-btn:active { transform: scale(0.9); }

        /* Dynamic Island Navigation */
        .dynamic-island {
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            background: rgba(20, 20, 20, 0.7); backdrop-filter: blur(30px) saturate(200%); -webkit-backdrop-filter: blur(30px) saturate(200%);
            border: 0.5px solid rgba(255,255,255,0.15); border-radius: 100px;
            display: flex; padding: 6px; gap: 4px; box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            z-index: 100;
        }
        .nav-indicator {
            position: absolute; top: 6px; left: 6px; width: 44px; height: 44px;
            background: rgba(255,255,255,0.12); border-radius: 50%;
            transition: transform 0.4s var(--bezier); z-index: 0; pointer-events: none;
        }
        .nav-btn {
            width: 44px; height: 44px; border-radius: 50%; border: none; background: transparent;
            color: var(--text-tertiary); display: flex; align-items: center; justify-content: center;
            font-size: 18px; cursor: pointer; position: relative; z-index: 1;
            transition: color 0.3s var(--bezier), transform 0.2s var(--bezier);
        }
        .nav-btn:active { transform: scale(0.85); }
        .nav-btn.nav-active { color: #fff; }

        /* Global Loader (VisionOS Style) */
        .loader-overlay {
            position: fixed; inset: 0; z-index: 9999;
            background: rgba(0,0,0,0.4); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            display: flex; flex-col; align-items: center; justify-content: center; flex-direction: column;
            opacity: 0; pointer-events: none; transition: opacity 0.4s var(--bezier);
        }
        .loader-overlay.active { opacity: 1; pointer-events: auto; }
        .spinner {
            width: 48px; height: 48px; border-radius: 50%;
            background: conic-gradient(from 0deg, transparent 0%, var(--accent) 100%);
            mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 0);
            -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 0);
            animation: spin 1s linear infinite; margin-bottom: 24px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .loader-title { font-size: 17px; font-weight: 600; color: #fff; margin-bottom: 8px; }
        .loader-text { font-size: 13px; color: var(--text-secondary); text-align: center; line-height: 1.4; max-width: 250px; }

        /* Print Override */
        .pdf-page { background: #000; color: #fff; }
    </style>
</head>
<body>

    <div class="ambient-bg">
        <div class="orb blue"></div>
        <div class="orb green"></div>
    </div>

    <!-- Global Loader -->
    <div id="global-loader" class="loader-overlay hidden">
        <div class="spinner"></div>
        <div id="loader-title" class="loader-title">Processing Document...</div>
        <div id="loader-text" class="loader-text">Please wait. AI is analyzing the data.</div>
    </div>

    <div class="app-container">
        
        <header class="header" data-html2canvas-ignore>
            <h1>Kaparsh</h1>
            <button id="download-btn" onclick="downloadNotes()" class="icon-btn hidden">
                <i class="fa-solid fa-arrow-down"></i>
            </button>
        </header>

        <main style="padding-top: 16px;">
            
            <!-- Tab: Home -->
            <div id="tab-home" class="tab-pane flex flex-col">
                <div class="glass-card" style="margin-bottom: 24px; border: none; background: radial-gradient(circle at top left, rgba(88, 101, 242, 0.2), transparent 70%), var(--surface-color);">
                    <h2 class="large-title">Chapter<br>Analysis</h2>
                    <p class="sub-title" style="margin-bottom: 0;">Upload a textbook chapter for deep-extraction notes, formulas, and intelligence.</p>
                </div>

                <label id="drop-zone" for="file-upload" class="upload-zone">
                    <div class="upload-icons">
                        <i class="fa-solid fa-file-pdf"></i>
                        <i class="fa-solid fa-image"></i>
                    </div>
                    <div id="file-name" class="upload-text">Tap to Upload File</div>
                    <input type="file" id="file-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
                </label>

                <button id="analyze-btn" class="btn-primary">
                    Generate Intelligence
                    <i class="fa-solid fa-wand-magic-sparkles" style="font-size: 14px;"></i>
                </button>
            </div>

            <!-- Tab: Notes -->
            <div id="tab-summary" class="tab-pane hidden pdf-page flex flex-col">
                <div class="section-header">
                    <span>Notes Feed</span>
                    <span id="topic-count" class="badge">0 Topics</span>
                </div>
                <div id="topics-grid" class="glass-card" style="padding: 24px 20px;">
                    <div style="text-align: center; padding: 40px 0; opacity: 0.5;">
                        <i class="fa-solid fa-layer-group" style="font-size: 32px; margin-bottom: 12px;"></i>
                        <p style="font-size: 14px; font-weight: 500;">Your digital notebook is empty.</p>
                    </div>
                </div>
            </div>

            <!-- Tab: Schedule -->
            <div id="tab-schedule" class="tab-pane hidden flex flex-col">
                <div id="schedule-setup">
                    <div style="text-align: center; margin-bottom: 32px;">
                        <div style="width: 64px; height: 64px; border-radius: 50%; background: rgba(88,101,242,0.1); border: 1px solid rgba(88,101,242,0.3); display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; font-size: 24px; color: var(--accent);">
                            <i class="fa-solid fa-calendar-day"></i>
                        </div>
                        <h2 class="large-title" style="font-size: 24px;">Master Planner</h2>
                        <p class="sub-title" style="margin-bottom: 0;">Set parameters and upload syllabus.</p>
                    </div>
                    
                    <div class="input-group">
                        <label>Target Exam Date</label>
                        <input type="date" id="schedule-exam-date">
                    </div>
                    <div class="input-group" style="margin-bottom: 24px;">
                        <label>Daily Study Hours</label>
                        <input type="number" id="schedule-study-hours" value="2" min="1" max="16" style="width: 60px;">
                    </div>

                    <label id="syllabus-drop-zone" for="syllabus-upload" class="upload-zone" style="padding: 30px 20px;">
                        <div class="upload-icons" style="margin-bottom: 8px;"><i class="fa-solid fa-book" style="color: var(--accent);"></i></div>
                        <div id="syllabus-file-name" class="upload-text" style="font-size: 14px;">Upload Syllabus (Optional)</div>
                        <input type="file" id="syllabus-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
                    </label>

                    <div id="subject-selector-area" class="hidden" style="margin-bottom: 24px;">
                        <p style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 12px;">Select Subjects to Study</p>
                        <div id="subject-checkboxes" class="checkbox-list"></div>
                    </div>

                    <button onclick="generateSchedule()" class="btn-primary">Build Master Plan</button>
                </div>
                
                <div id="schedule-result" class="hidden flex-col">
                    <div class="section-header">Timeline</div>
                    <div class="timeline" id="schedule-timeline-container"></div>
                </div>
            </div>

            <!-- Tab: Quiz -->
            <div id="tab-quiz" class="tab-pane hidden flex flex-col">
                <div id="quiz-setup" style="text-align: center; padding: 40px 0;">
                    <div style="width: 80px; height: 80px; border-radius: 50%; background: rgba(50, 215, 75, 0.1); border: 1px solid rgba(50, 215, 75, 0.3); display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; font-size: 32px; color: var(--success);">
                        <i class="fa-solid fa-gamepad"></i>
                    </div>
                    <h2 class="large-title" style="font-size: 28px;">Dynamic Quiz</h2>
                    <p class="sub-title">Test mastery of the analyzed chapter.</p>
                    <button onclick="generateQuiz()" class="btn-primary success-style" style="margin-top: 16px;">Start Knowledge Check</button>
                </div>
                <div id="quiz-result" class="hidden flex-col">
                    <div id="quiz-questions-container"></div>
                    <button onclick="checkAnswers()" class="btn-primary" style="margin-top: 24px;">Submit Answers</button>
                    <div id="quiz-score-area"></div>
                </div>
            </div>

            <!-- Tab: Doubts -->
            <div id="tab-doubts" class="tab-pane hidden">
                <div class="chat-container">
                    <div class="section-header" style="margin-bottom: 24px;">Study Assistant</div>
                    <div id="chat-history" class="chat-history">
                        <div class="chat-bubble chat-ai">Hi there! Ask me anything about the chapter you just analyzed.</div>
                    </div>
                </div>
                <div class="chat-input-wrapper" data-html2canvas-ignore>
                    <div class="chat-input-box">
                        <input type="text" id="doubt-input" placeholder="Message Kaparsh...">
                        <button onclick="sendDoubt()" class="chat-send-btn"><i class="fa-solid fa-arrow-up"></i></button>
                    </div>
                </div>
            </div>
        </main>

        <!-- Navigation (Dynamic Island) -->
        <nav class="dynamic-island" data-html2canvas-ignore>
            <div class="nav-indicator" id="nav-indicator"></div>
            <button class="nav-btn nav-active" data-target="tab-home" onclick="switchTab('tab-home', 0)"><i class="fa-solid fa-house"></i></button>
            <button class="nav-btn" data-target="tab-summary" onclick="switchTab('tab-summary', 1)"><i class="fa-solid fa-layer-group"></i></button>
            <button class="nav-btn" data-target="tab-schedule" onclick="switchTab('tab-schedule', 2)"><i class="fa-solid fa-calendar-day"></i></button>
            <button class="nav-btn" data-target="tab-quiz" onclick="switchTab('tab-quiz', 3)"><i class="fa-solid fa-gamepad"></i></button>
            <button class="nav-btn" data-target="tab-doubts" onclick="switchTab('tab-doubts', 4)"><i class="fa-solid fa-comment-dots"></i></button>
        </nav>

    </div>

    <script>
        const AppState = {
            extractedText: "", topics: null, schedule: null, quiz: null,
            file: null, syllabusFile: null, syllabusContextText: "", globalBannedTerms: [] 
        };

        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 14);
        document.getElementById('schedule-exam-date').valueAsDate = tomorrow;

        // UI Logic
        const navBtns = document.querySelectorAll('.nav-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        const indicator = document.getElementById('nav-indicator');

        function switchTab(targetId, index) {
            if (targetId === 'tab-summary' || targetId === 'tab-quiz' || targetId === 'tab-doubts') {
                if (!AppState.topics) return alert("Upload and analyze a Chapter first!");
            }
            
            navBtns.forEach(btn => btn.classList.remove('nav-active'));
            navBtns[index].classList.add('nav-active');
            indicator.style.transform = `translateX(${index * 48}px)`;

            tabPanes.forEach(pane => { pane.classList.add('hidden'); pane.classList.remove('flex'); });
            const target = document.getElementById(targetId);
            target.classList.remove('hidden');
            target.classList.add('flex');
            window.scrollTo(0, 0);
        }

        const toggleLoader = (show, title = 'Processing Document...', text = 'Please wait. AI is analyzing the data.') => {
            document.getElementById('loader-title').innerText = title;
            document.getElementById('loader-text').innerText = text;
            const loader = document.getElementById('global-loader');
            
            if (show) {
                loader.classList.remove('hidden');
                // Small delay to allow CSS block rendering before active class
                setTimeout(() => loader.classList.add('active'), 10);
            } else {
                loader.classList.remove('active');
                setTimeout(() => loader.classList.add('hidden'), 400); // Wait for transition
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
                fileNameDisplay.style.color = 'var(--accent)';
            }
        });
        
        const syllabusInput = document.getElementById('syllabus-upload');
        const syllabusNameDisplay = document.getElementById('syllabus-file-name');
        syllabusInput.addEventListener('change', async () => {
            if (syllabusInput.files.length > 0) {
                AppState.syllabusFile = syllabusInput.files[0];
                syllabusNameDisplay.textContent = AppState.syllabusFile.name;
                syllabusNameDisplay.style.color = 'var(--success)';
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
                alert("Auto-detection note: " + err.message + ". You can still generate your master plan normally.");
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
            toggleLoader(true, 'Analyzing Chapter...', 'Reading document via server...');

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
            document.getElementById('topic-count').innerText = `${AppState.topics.length} Topics`;
            
            if (AppState.topics.some(t => t.loaded)) {
                document.getElementById('download-btn').classList.remove('hidden');
            }
            
            grid.innerHTML = AppState.topics.map((t) => {
                if (!t.loaded) {
                    return `
                    <div class="note-card" style="opacity: 0.5; animation: fadeUp 1s infinite alternate;">
                        <div style="height: 20px; background: var(--surface-border); border-radius: 4px; width: 60%; margin-bottom: 12px;"></div>
                        <div style="height: 12px; background: var(--surface-border); border-radius: 4px; width: 80%;"></div>
                    </div>`;
                }

                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.definitions.map(d => `
                            <div class="def-box">
                                <span>${d.term}:</span><p>${d.definition}</p>
                            </div>
                        `).join('')}
                    </div>` : '';

                const formulasHtml = (t.formulas && t.formulas.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.formulas.map(f => `
                            <div class="formula-box">
                                <div class="eq">${f.equation}</div>
                                <div class="meaning">${f.meaning}</div>
                            </div>
                        `).join('')}
                    </div>` : '';

                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.derivations.map(d => `
                            <div class="deriv-box">
                                <h5>${d.title}</h5>
                                <pre>${d.content}</pre>
                            </div>
                        `).join('')}
                    </div>` : '';

                const notesList = (t.notes || []).map(n => `<li>${n}</li>`).join('');

                return `
                <div class="note-card">
                    <div class="note-header">
                        <div class="note-title">${t.title}</div>
                        <div class="badge">${t.priority || 'MED'}</div>
                    </div>
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    ${notesList ? `<ul class="note-list" style="margin-top:16px;">${notesList}</ul>` : ''}
                </div>`;
            }).join('');
        }

        function downloadNotes() {
            const element = document.getElementById('tab-summary');
            const opt = {
                margin: 0.5,
                filename: 'Kaparsh_Notes.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2, useCORS: true, backgroundColor: '#000000' },
                jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
            };
            html2pdf().set(opt).from(element).save();
        }

        // Schedule Logic
        async function generateSchedule() {
            const examDate = document.getElementById('schedule-exam-date').value;
            const studyHours = document.getElementById('schedule-study-hours').value;
            
            if (!examDate) return alert("Please set your Exam Date first.");

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
            
            container.innerHTML = AppState.schedule.map((day) => `
                <div class="timeline-item">
                    <div class="timeline-dot">${day.day || '-'}</div>
                    <div class="timeline-content">
                        <div class="tl-header">
                            <span class="tl-date">${(day.date || '').split('-').slice(1).join('/') || 'Day'}</span>
                            <span class="tl-hours"><i class="fa-regular fa-clock"></i> ${day.total_hours_today || 0}h</span>
                        </div>
                        <div class="tl-focus">${day.focus_area || 'Study Block'}</div>
                        <div>
                            ${(day.topics || []).map(t => `
                                <div class="tl-topic">
                                    <span class="tl-topic-name">${t.name || 'Topic'}</span>
                                    <span class="tl-topic-time">${t.estimated_minutes || 30} min</span>
                                </div>
                            `).join('')}
                        </div>
                        ${day.actionable_advice ? `<div class="tl-advice">${day.actionable_advice}</div>` : ''}
                    </div>
                </div>
            `).join('');
        }

        // Quiz Logic
        async function generateQuiz() {
            toggleLoader(true, 'Generating Exam...', 'Building adaptive questions...');
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

            container.innerHTML = AppState.quiz.map((q, index) => `
                <div id="qcard-${index}" class="glass-card q-card">
                    <div class="q-label">QUESTION ${index + 1}</div>
                    <div class="q-text">${q.question || 'Missing question'}</div>
                    <div>
                        ${(q.options || []).map(opt => `
                            <label class="quiz-opt-label">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}">
                                <span>${opt}</span>
                            </label>
                        `).join('')}
                    </div>
                    <div id="result-${index}"></div>
                </div>
            `).join('');
        }

        function checkAnswers() {
            let score = 0;
            AppState.quiz.forEach((q, index) => {
                const selected = document.querySelector(`input[name="question-${index}"]:checked`);
                const resultDiv = document.getElementById(`result-${index}`);
                const card = document.getElementById(`qcard-${index}`);
                
                if (!selected) {
                    resultDiv.innerHTML = `<div style="color: var(--danger); font-size: 13px; font-weight: 600; margin-top: 12px;"><i class="fa-solid fa-circle-exclamation"></i> Answer required</div>`;
                    return;
                }
                
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box correct">
                            <strong style="color: var(--success); display: block; margin-bottom: 4px;">Correct</strong>
                            <span style="color: rgba(255,255,255,0.8);">${q.explanation}</span>
                        </div>`;
                    card.style.borderColor = 'rgba(50, 215, 75, 0.4)';
                } else {
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box wrong">
                            <strong style="color: var(--danger); display: block; margin-bottom: 4px;">Incorrect</strong>
                            <div style="font-size: 12px; margin-bottom: 6px;">Correct: <strong>${q.correct_answer}</strong></div>
                            <span style="color: rgba(255,255,255,0.8);">${q.explanation}</span>
                        </div>`;
                    card.style.borderColor = 'rgba(255, 69, 58, 0.4)';
                }
            });
            
            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="glass-card" style="text-align: center; margin-top: 24px; padding: 40px 20px;">
                        <p style="font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px;">Final Score</p>
                        <div style="font-size: 48px; font-weight: 800; color: #fff; line-height: 1;">${score} <span style="font-size: 24px; color: var(--text-tertiary);">/ ${AppState.quiz.length}</span></div>
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
            if (!AppState.extractedText) return alert("Please upload and process a Chapter first to ask doubts.");

            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML += `<div class="chat-bubble chat-user">${question}</div>`;
            input.value = '';
            
            // Scroll to bottom
            const container = document.querySelector('.chat-container');
            chatHistory.scrollTop = chatHistory.scrollHeight;

            const loaderId = 'loader-' + Date.now();
            chatHistory.innerHTML += `
                <div id="${loaderId}" class="chat-bubble chat-ai" style="padding: 16px;">
                    <div class="typing-dots"><span></span><span></span><span></span></div>
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
                
                const formattedAnswer = data.answer.replace(/\n/g, '<br>');
                chatHistory.innerHTML += `<div class="chat-bubble chat-ai">${formattedAnswer}</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
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