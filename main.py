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
# MOBILE NATIVE UI INJECTION (OPTIMIZED FOR 6.7" SMARTPHONES)
# PREMIUM APPLE / GEMINI VIBRANT EXPRESSIVE DESIGN
# ==============================================================================

KAPARSH_FRONTEND = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#0A0A0F">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Kaparsh</title>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

    <style>
        /* APPLE HIG / GEMINI VIBRANT PREMIUM DESIGN SYSTEM */
        :root {
            --bg-color: #0A0A0F; /* Deep premium charcoal/blue instead of #000 */
            --surface-color: rgba(255, 255, 255, 0.04);
            --surface-border: rgba(255, 255, 255, 0.1);
            
            /* Vibrant gradients and glowing accents */
            --accent: #7C3AED; 
            --accent-grad: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            --accent-glow: rgba(124, 58, 237, 0.4);
            --success: #10B981;
            --success-grad: linear-gradient(135deg, #059669 0%, #10B981 100%);
            --danger: #F43F5E;
            
            /* Brighter, warmer typography */
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.7);
            --text-tertiary: rgba(255, 255, 255, 0.4);
            --bezier: cubic-bezier(0.25, 1, 0.5, 1);
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
            overflow-x: hidden;
            overscroll-behavior-y: none; /* Prevent pull-to-refresh bounce */
        }

        /* Hide scrollbars */
        ::-webkit-scrollbar { display: none; }
        * { -ms-overflow-style: none; scrollbar-width: none; }

        /* Expressive, Vivid Liquid Background Orbs (Gemini/Siri Style) */
        .ambient-bg {
            position: fixed; inset: 0; z-index: -1;
            overflow: hidden; pointer-events: none;
            background: var(--bg-color);
        }
        .orb {
            position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.4;
            animation: float 20s infinite alternate ease-in-out;
            mix-blend-mode: screen;
        }
        .orb.one { width: 85vw; height: 85vw; background: #4F46E5; top: -15%; left: -25%; } /* Indigo */
        .orb.two { width: 75vw; height: 75vw; background: #EC4899; bottom: -5%; right: -25%; animation-delay: -6s; } /* Magenta/Pink */
        .orb.three { width: 65vw; height: 65vw; background: #06B6D4; top: 35%; left: 35%; animation-delay: -12s; opacity: 0.35; } /* Cyan */
        
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(15vw, 15vh) scale(1.15); }
            100% { transform: translate(-10vw, 20vh) scale(1.05); }
        }

        /* Mobile Container Layout constraints for 6.7" */
        .app-container {
            width: 100%;
            max-width: 430px; /* iPhone 15 Pro Max width */
            margin: 0 auto;
            min-height: 100vh;
            position: relative;
            padding: env(safe-area-inset-top) 16px calc(130px + env(safe-area-inset-bottom)) 16px;
            display: flex; flex-direction: column;
        }
        
        /* Header */
        .header {
            position: sticky; top: env(safe-area-inset-top); z-index: 40;
            padding: 12px 0; margin-bottom: 12px;
            display: flex; justify-content: space-between; align-items: center;
            background: transparent;
        }
        .header h1 { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; 
                     background: linear-gradient(135deg, #FFF, #E0E0E0);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        .icon-btn {
            width: 36px; height: 36px; border-radius: 50%; border: none;
            background: rgba(255, 255, 255, 0.08); color: var(--text-primary);
            display: flex; align-items: center; justify-content: center; font-size: 14px;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.12); cursor: pointer;
            transition: all 0.2s var(--bezier); box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .icon-btn:active { transform: scale(0.9); background: rgba(255,255,255,0.15); }

        /* Typography */
        .large-title { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; line-height: 1.2; }
        .sub-title { font-size: 15px; color: var(--text-secondary); line-height: 1.4; font-weight: 400; margin-bottom: 20px; }
        .section-header { font-size: 18px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; color: #FFF; text-shadow: 0 2px 8px rgba(0,0,0,0.5); }

        /* Utilities */
        .hidden { display: none !important; }
        .flex { display: flex !important; }
        .tab-pane { flex: 1; animation: fadeScale 0.4s var(--bezier); display: flex; flex-direction: column; }
        
        @keyframes fadeScale {
            from { opacity: 0; transform: scale(0.97) translateY(10px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        /* Vivid Glass Cards */
        .glass-card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
            backdrop-filter: blur(40px) saturate(200%); -webkit-backdrop-filter: blur(40px) saturate(200%);
            border: 1px solid var(--surface-border);
            border-radius: 24px; padding: 20px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
            margin-bottom: 16px; position: relative; overflow: hidden;
        }

        /* Upload Zones */
        .upload-zone {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 32px 16px; border-radius: 24px;
            background: rgba(255, 255, 255, 0.03);
            border: 1.5px dashed rgba(255, 255, 255, 0.2);
            cursor: pointer; transition: all 0.2s var(--bezier);
            text-align: center; margin-bottom: 20px;
        }
        .upload-zone:active, .upload-zone:hover { 
            transform: scale(0.97); 
            background: rgba(255, 255, 255, 0.08); 
            border-color: rgba(255, 255, 255, 0.4); 
            box-shadow: 0 0 20px rgba(255,255,255,0.05);
        }
        .upload-zone input[type="file"] { position: absolute; opacity: 0; width: 0; height: 0; }
        .upload-icons { display: flex; gap: 12px; margin-bottom: 12px; }
        .upload-icons i { font-size: 26px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.4)); }
        .upload-text { font-size: 15px; font-weight: 600; color: #FFF; }

        /* Premium Radiant Button */
        .btn-primary {
            width: 100%; padding: 16px; border-radius: 16px;
            background: var(--accent-grad); color: #fff; text-shadow: 0 1px 4px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1); font-size: 16px; font-weight: 600; letter-spacing: 0.3px;
            cursor: pointer; transition: all 0.2s var(--bezier);
            box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35), inset 0 1px 1px rgba(255,255,255,0.2);
            display: flex; justify-content: center; align-items: center; gap: 8px;
        }
        .btn-primary:active { transform: scale(0.96); opacity: 0.9; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3); }
        .btn-success { 
            background: var(--success-grad); color: #fff; 
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3), inset 0 1px 1px rgba(255,255,255,0.2); 
        }

        /* Native Inputs */
        .input-group {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; padding: 14px 18px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .input-group label { font-size: 14px; color: var(--text-secondary); font-weight: 600; }
        .input-group input { 
            background: transparent; border: none; outline: none; color: #fff; 
            font-size: 16px; font-weight: 700; text-align: right; 
            -webkit-appearance: none; appearance: none;
        }
        .input-group input[type="date"] { color: var(--success); color-scheme: dark; }

        /* Expressive Checkboxes */
        .checkbox-list { display: flex; flex-direction: column; gap: 10px; max-height: 240px; overflow-y: auto; margin-bottom: 24px; }
        .custom-checkbox {
            display: flex; align-items: center; gap: 12px; padding: 14px 16px;
            background: rgba(255,255,255,0.04); border: 1px solid var(--surface-border);
            border-radius: 16px; transition: 0.2s var(--bezier); cursor: pointer;
        }
        .custom-checkbox:active { transform: scale(0.98); }
        .custom-checkbox input { display: none; }
        .checker {
            width: 22px; height: 22px; border-radius: 50%; border: 2px solid var(--text-tertiary);
            display: flex; align-items: center; justify-content: center; transition: 0.2s var(--bezier);
            background: rgba(0,0,0,0.3);
        }
        .checker i { font-size: 11px; color: #fff; opacity: 0; transform: scale(0.5); transition: 0.3s var(--bezier); }
        .custom-checkbox input:checked + .checker { background: var(--accent-grad); border-color: transparent; box-shadow: 0 2px 8px rgba(124, 58, 237, 0.4); }
        .custom-checkbox input:checked + .checker i { opacity: 1; transform: scale(1); }
        .checkbox-label { font-size: 15px; font-weight: 500; flex: 1; color: var(--text-primary); }
        .custom-checkbox:has(input:checked) { background: rgba(255,255,255,0.08); border-color: rgba(124, 58, 237, 0.4); }

        /* Markdown System Styling (Optimized Legibility) */
        .md-content { line-height: 1.6; word-wrap: break-word; color: #fff; }
        .md-content p { margin-bottom: 12px; }
        .md-content p:last-child { margin-bottom: 0; }
        .md-content strong { color: #fff; font-weight: 700; background: linear-gradient(90deg, #fff, #d4d4d8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .md-content em { font-style: italic; color: var(--text-secondary); }
        .md-content ul { padding-left: 20px; margin-bottom: 14px; list-style-type: none; }
        .md-content ol { padding-left: 20px; margin-bottom: 14px; }
        .md-content li { margin-bottom: 6px; padding-left: 8px; position: relative; }
        .md-content ul li::before { content: '•'; position: absolute; left: -14px; color: var(--accent); font-weight: bold; font-size: 1.2em; top: -2px; }
        .md-content code { background: rgba(255,255,255,0.12); padding: 2px 6px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 0.9em; border: 1px solid rgba(255,255,255,0.05); }
        .md-content pre { background: rgba(0,0,0,0.6); padding: 14px; border-radius: 14px; overflow-x: auto; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.08); }
        .md-content pre code { background: transparent; padding: 0; font-size: 13px; border: none; }
        .md-content h1, .md-content h2, .md-content h3, .md-content h4 { font-weight: 700; margin-top: 18px; margin-bottom: 10px; color: #fff; line-height: 1.2; text-shadow: 0 2px 10px rgba(124, 58, 237, 0.2); }
        .md-content h3 { color: #EC4899; }
        .md-content blockquote { border-left: 3px solid var(--accent); padding-left: 14px; margin-left: 0; color: var(--text-secondary); background: linear-gradient(90deg, rgba(124,58,237,0.1), transparent); padding-top: 8px; padding-bottom: 8px; border-radius: 0 8px 8px 0; }
        .katex { color: #E2E8F0; font-size: 1.05em; } 
        .katex-display { margin: 12px 0; overflow-x: auto; overflow-y: hidden; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 12px; }

        /* Rich Notes UI */
        .badge {
            font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;
            padding: 4px 10px; border-radius: 8px;
            background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05));
            color: #FFF; border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .note-card { border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 24px; margin-bottom: 24px; }
        .note-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .note-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
        .note-title { font-size: 18px; font-weight: 700; color: #fff; line-height: 1.3; max-width: 80%; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        
        .def-box {
            background: linear-gradient(90deg, rgba(124, 58, 237, 0.15), rgba(79, 70, 229, 0.05)); 
            border-left: 4px solid var(--accent);
            padding: 12px 14px; border-radius: 0 12px 12px 0; margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .def-box span.term-title { font-weight: 700; color: #A78BFA; margin-right: 6px; font-size: 14px; text-shadow: 0 1px 2px rgba(0,0,0,0.4); }
        .def-box .def-text { font-size: 14px; color: rgba(255,255,255,0.9); display: inline; line-height: 1.5; }
        .def-box .def-text p { display: inline; margin: 0; }

        .formula-box {
            background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1);
            padding: 16px; border-radius: 16px; text-align: center; margin-bottom: 12px;
            box-shadow: inset 0 2px 12px rgba(0,0,0,0.3);
        }
        .formula-box .eq { font-family: ui-monospace, monospace; font-size: 18px; font-weight: 800; color: #34D399; margin-bottom: 4px; text-shadow: 0 2px 10px rgba(16, 185, 129, 0.3); }
        .formula-box .meaning { font-size: 12px; color: var(--text-secondary); font-weight: 500; }

        .deriv-box {
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
            padding: 14px; border-radius: 16px; margin-bottom: 12px;
        }
        .deriv-box h5 { font-size: 14px; color: #60A5FA; margin-bottom: 8px; font-weight: 700; }
        .deriv-box .deriv-content { font-family: ui-monospace, monospace; font-size: 13px; color: rgba(255,255,255,0.8); white-space: pre-wrap; line-height: 1.5; }

        /* Timeline (Vibrant Schedule) */
        .timeline { position: relative; padding-left: 28px; margin-top: 16px; margin-left: 12px; }
        .timeline::before {
            content: ''; position: absolute; left: 0px; top: 0; bottom: 0; width: 2px;
            background: linear-gradient(to bottom, var(--accent) 0%, #EC4899 50%, rgba(255,255,255,0.05) 100%);
            border-radius: 2px;
        }
        .timeline-item { position: relative; margin-bottom: 32px; }
        .timeline-dot {
            position: absolute; left: -39px; top: 12px; width: 24px; height: 24px;
            border-radius: 50%; background: var(--bg-color); border: 3px solid var(--accent);
            display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800;
            box-shadow: 0 0 12px var(--accent-glow); color: #fff; z-index: 2;
        }
        .timeline-content {
            background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 18px;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        .tl-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .tl-date { color: #34D399; font-size: 13px; font-weight: 800; letter-spacing: 0.5px; }
        .tl-hours { color: var(--text-secondary); font-size: 12px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
        .tl-focus { font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #FFF; }
        .tl-topic {
            background: rgba(0,0,0,0.4); padding: 10px 14px; border-radius: 12px;
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .tl-topic-name { font-size: 13px; font-weight: 600; color: #E2E8F0; }
        .tl-topic-time { font-size: 11px; color: #fff; font-weight: 700; background: var(--accent-grad); padding: 4px 8px; border-radius: 8px; box-shadow: 0 2px 6px rgba(124, 58, 237, 0.4); }
        .tl-advice { font-size: 13px; color: rgba(255,255,255,0.8); margin-top: 14px; padding-left: 10px; border-left: 3px solid #EC4899; line-height: 1.5; }

        /* Quiz */
        .q-card { margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.12); }
        .q-label { font-size: 12px; font-weight: 800; color: #A78BFA; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase; }
        .q-text { font-size: 17px; font-weight: 600; line-height: 1.5; margin-bottom: 20px; color: #FFF; }
        
        .quiz-opt-label {
            display: flex; align-items: center; padding: 14px 16px; margin-bottom: 12px;
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; font-size: 15px; font-weight: 500; line-height: 1.4;
            transition: all 0.2s var(--bezier); cursor: pointer; color: #E2E8F0;
        }
        .quiz-opt-label input { display: none; }
        .quiz-opt-label:active { transform: scale(0.97); }
        .quiz-opt-label:has(input:checked) {
            background: rgba(124, 58, 237, 0.15); border-color: #8B5CF6;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2); color: #fff; font-weight: 600;
        }
        .quiz-res-box { padding: 14px; border-radius: 14px; margin-top: 16px; font-size: 14px; line-height: 1.5; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
        .quiz-res-box.correct { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2); }
        .quiz-res-box.wrong { background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.4); box-shadow: 0 4px 16px rgba(244, 63, 94, 0.2); }

        /* Vibrant Chat UI */
        .chat-pane { flex: 1; display: flex; flex-direction: column; overflow: hidden; margin: -16px; padding: 16px; padding-bottom: 80px; }
        .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; padding-bottom: 24px; }
        
        .chat-bubble { max-width: 88%; padding: 14px 18px; border-radius: 22px; font-size: 15px; line-height: 1.5; animation: fadeScale 0.3s var(--bezier); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        
        .chat-ai {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            border: 1px solid rgba(255,255,255,0.1); color: #FFF;
            align-self: flex-start; border-bottom-left-radius: 6px; 
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        }
        .chat-user { 
            background: var(--accent-grad); color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            align-self: flex-end; border-bottom-right-radius: 6px; 
            box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3); border: 1px solid rgba(255,255,255,0.15);
        }
        
        .typing-dots { display: flex; gap: 5px; padding: 6px 4px; }
        .typing-dots span { width: 7px; height: 7px; background: rgba(255,255,255,0.8); border-radius: 50%; animation: typeDot 1.4s infinite ease-in-out both; }
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typeDot { 0%, 80%, 100% { transform: scale(0); opacity: 0.3; } 40% { transform: scale(1); opacity: 1; } }

        /* Premium Input Box */
        .chat-input-wrapper {
            position: fixed; bottom: calc(112px + env(safe-area-inset-bottom)); 
            left: 50%; transform: translateX(-50%);
            width: 100%; max-width: 430px; padding: 0 16px; z-index: 40;
        }
        .chat-input-box {
            background: rgba(15, 15, 20, 0.75); backdrop-filter: blur(30px) saturate(200%); -webkit-backdrop-filter: blur(30px) saturate(200%);
            border: 1px solid rgba(255,255,255,0.15); border-radius: 100px;
            display: flex; align-items: center; padding: 8px 8px 8px 20px;
            box-shadow: 0 16px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .chat-input-box input {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 16px; font-weight: 500;
        }
        .chat-input-box input::placeholder { color: var(--text-tertiary); font-weight: 400; }
        .chat-send-btn {
            width: 38px; height: 38px; border-radius: 50%; background: var(--accent-grad);
            border: none; color: #fff; display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: 0.2s var(--bezier); flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4); font-size: 15px;
        }
        .chat-send-btn:active { transform: scale(0.9); }

        /* EXPRESSIVE Dynamic Island Navigation */
        .dynamic-island {
            position: fixed; bottom: calc(24px + env(safe-area-inset-bottom)); 
            left: 50%; transform: translateX(-50%);
            background: rgba(15, 15, 20, 0.7); backdrop-filter: blur(40px) saturate(200%); -webkit-backdrop-filter: blur(40px) saturate(200%);
            border: 1px solid rgba(255,255,255,0.15); border-radius: 100px;
            display: flex; padding: 10px; gap: 8px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
            z-index: 100;
        }
        .nav-indicator {
            position: absolute; top: 10px; left: 10px; width: 56px; height: 56px;
            background: rgba(255,255,255,0.15); border-radius: 50%;
            transition: transform 0.4s var(--bezier); z-index: 0; pointer-events: none;
            box-shadow: 0 0 12px rgba(255,255,255,0.1);
        }
        .nav-btn {
            width: 56px; height: 56px;
            border-radius: 50%; border: none; background: transparent;
            color: rgba(255,255,255,0.4); display: flex; align-items: center; justify-content: center;
            font-size: 22px; cursor: pointer; position: relative; z-index: 1;
            transition: all 0.3s var(--bezier);
        }
        .nav-btn:active { transform: scale(0.85); }
        .nav-btn.nav-active { color: #fff; text-shadow: 0 0 16px rgba(255,255,255,0.6); transform: scale(1.05); }

        /* Global Loader (Brighter and Premium) */
        .loader-overlay {
            position: fixed; inset: 0; z-index: 9999;
            background: rgba(10,10,15,0.7); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            opacity: 0; pointer-events: none; transition: opacity 0.3s var(--bezier);
        }
        .loader-overlay.active { opacity: 1; pointer-events: auto; }
        .spinner {
            width: 52px; height: 52px; border-radius: 50%;
            background: conic-gradient(from 0deg, transparent 0%, #7C3AED 50%, #EC4899 100%);
            mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 0);
            -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 0);
            animation: spin 1s linear infinite; margin-bottom: 24px; filter: drop-shadow(0 0 8px rgba(124, 58, 237, 0.5));
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .loader-title { font-size: 19px; font-weight: 700; color: #fff; margin-bottom: 8px; letter-spacing: 0.3px; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }
        .loader-text { font-size: 14px; color: rgba(255,255,255,0.7); text-align: center; line-height: 1.5; max-width: 260px; }
        
        .pdf-page { background: var(--bg-color); color: #fff; }
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
            
        <!-- Tab: Home -->
        <div id="tab-home" class="tab-pane flex">
            <div class="glass-card" style="margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.15); background: radial-gradient(circle at top left, rgba(124, 58, 237, 0.35), transparent 80%), linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));">
                <h2 class="large-title">Chapter<br>Analysis</h2>
                <p class="sub-title" style="margin-bottom: 0;">Upload a textbook chapter for deep-extraction notes, formulas, and intelligence.</p>
            </div>

            <label id="drop-zone" for="file-upload" class="upload-zone">
                <div class="upload-icons">
                    <i class="fa-solid fa-file-pdf" style="color: #A78BFA;"></i>
                    <i class="fa-solid fa-image" style="color: #34D399;"></i>
                </div>
                <div id="file-name" class="upload-text">Tap to Upload File</div>
                <input type="file" id="file-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
            </label>

            <button id="analyze-btn" class="btn-primary">
                Generate Intelligence
                <i class="fa-solid fa-wand-magic-sparkles" style="font-size: 15px; margin-left: 4px;"></i>
            </button>
        </div>

        <!-- Tab: Notes -->
        <div id="tab-summary" class="tab-pane hidden pdf-page">
            <div class="section-header">
                <span>Notes Feed</span>
                <span id="topic-count" class="badge">0 Topics</span>
            </div>
            <div id="topics-grid" class="glass-card" style="padding: 24px 18px;">
                <div style="text-align: center; padding: 40px 0; opacity: 0.6;">
                    <i class="fa-solid fa-layer-group" style="font-size: 36px; margin-bottom: 16px; color: rgba(255,255,255,0.5);"></i>
                    <p style="font-size: 15px; font-weight: 500;">Your digital notebook is empty.</p>
                </div>
            </div>
        </div>

        <!-- Tab: Schedule -->
        <div id="tab-schedule" class="tab-pane hidden">
            <div id="schedule-setup">
                <div style="text-align: center; margin-bottom: 28px;">
                    <div style="width: 64px; height: 64px; border-radius: 50%; background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(236, 72, 153, 0.2)); border: 1px solid rgba(255, 255, 255, 0.15); display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; font-size: 26px; color: #FFF; box-shadow: 0 8px 20px rgba(124, 58, 237, 0.25);">
                        <i class="fa-solid fa-calendar-day"></i>
                    </div>
                    <h2 class="large-title" style="font-size: 26px;">Master Planner</h2>
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

                <label id="syllabus-drop-zone" for="syllabus-upload" class="upload-zone" style="padding: 28px 16px;">
                    <div class="upload-icons" style="margin-bottom: 10px;"><i class="fa-solid fa-book" style="color: #60A5FA;"></i></div>
                    <div id="syllabus-file-name" class="upload-text" style="font-size: 15px;">Upload Syllabus (Optional)</div>
                    <input type="file" id="syllabus-upload" accept="application/pdf, image/png, image/jpeg, image/jpg">
                </label>

                <div id="subject-selector-area" class="hidden" style="margin-bottom: 24px;">
                    <p style="font-size: 12px; font-weight: 700; color: #A78BFA; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px;">Select Subjects to Study</p>
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
        <div id="tab-quiz" class="tab-pane hidden">
            <div id="quiz-setup" style="text-align: center; padding: 32px 0;">
                <div style="width: 68px; height: 68px; border-radius: 50%; background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2)); border: 1px solid rgba(16, 185, 129, 0.4); display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; font-size: 30px; color: #34D399; box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);">
                    <i class="fa-solid fa-gamepad"></i>
                </div>
                <h2 class="large-title" style="font-size: 26px;">Dynamic Quiz</h2>
                <p class="sub-title">Test mastery of the analyzed chapter.</p>
                <button onclick="generateQuiz()" class="btn-primary btn-success" style="margin-top: 24px;">Start Knowledge Check</button>
            </div>
            <div id="quiz-result" class="hidden flex-col">
                <div id="quiz-questions-container"></div>
                <button onclick="checkAnswers()" class="btn-primary" style="margin-top: 20px;">Submit Answers</button>
                <div id="quiz-score-area"></div>
            </div>
        </div>

        <!-- Tab: Doubts -->
        <div id="tab-doubts" class="tab-pane chat-pane hidden">
            <div class="section-header">Study Assistant</div>
            <div id="chat-history" class="chat-history">
                <div class="chat-bubble chat-ai md-content">Hi there! Ask me anything about the chapter you just analyzed.</div>
            </div>
            <div class="chat-input-wrapper" data-html2canvas-ignore>
                <div class="chat-input-box">
                    <input type="text" id="doubt-input" placeholder="Message Kaparsh...">
                    <button onclick="sendDoubt()" class="chat-send-btn"><i class="fa-solid fa-arrow-up"></i></button>
                </div>
            </div>
        </div>
        
    </div>

    <!-- Navigation (Dynamic Island) -->
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
                    errorColor: '#F43F5E'
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
            
            // Transform logic for nav bar (56px width + 8px gap = 64px spacing step)
            indicator.style.transform = `translateX(${index * 64}px)`;

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
                setTimeout(() => loader.classList.add('active'), 10);
            } else {
                loader.classList.remove('active');
                setTimeout(() => loader.classList.add('hidden'), 300);
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
                fileNameDisplay.style.color = '#A78BFA';
            }
        });
        
        const syllabusInput = document.getElementById('syllabus-upload');
        const syllabusNameDisplay = document.getElementById('syllabus-file-name');
        syllabusInput.addEventListener('change', async () => {
            if (syllabusInput.files.length > 0) {
                AppState.syllabusFile = syllabusInput.files[0];
                syllabusNameDisplay.textContent = AppState.syllabusFile.name;
                syllabusNameDisplay.style.color = '#34D399';
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
                    <div class="note-card" style="opacity: 0.5; animation: fadeScale 1s infinite alternate;">
                        <div style="height: 20px; background: rgba(255,255,255,0.1); border-radius: 6px; width: 60%; margin-bottom: 14px;"></div>
                        <div style="height: 14px; background: rgba(255,255,255,0.06); border-radius: 6px; width: 85%;"></div>
                    </div>`;
                }

                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div style="margin-top: 16px;">
                        ${t.definitions.map(d => {
                            const inlineDef = marked.parseInline(d.definition || '');
                            return `
                            <div class="def-box">
                                <span class="term-title">${d.term}:</span>
                                <div class="def-text md-content">${inlineDef}</div>
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
                                <div class="meaning md-content">${inlineMean}</div>
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
                            <div class="deriv-box">
                                <h5>${inlineTitle}</h5>
                                <div class="deriv-content md-content">${bodyContent}</div>
                            </div>
                            `;
                        }).join('')}
                    </div>` : '';

                const notesList = (t.notes || []).map(n => {
                    return `<li><div class="md-content" style="display:inline;">${marked.parseInline(n)}</div></li>`;
                }).join('');

                return `
                <div class="note-card">
                    <div class="note-header">
                        <div class="note-title">${t.title}</div>
                        <div class="badge">${t.priority || 'MED'}</div>
                    </div>
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    ${notesList ? `<ul class="note-list md-content" style="margin-top:16px;">${notesList}</ul>` : ''}
                </div>`;
            }).join('');

            // Allow elements to attach to DOM before triggering KaTeX rendering
            setTimeout(applyMath, 50);
        }

        function downloadNotes() {
            const element = document.getElementById('tab-summary');
            const opt = {
                margin: 0.5,
                filename: 'Kaparsh_Notes.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2, useCORS: true, backgroundColor: '#0A0A0F' },
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

            container.innerHTML = AppState.quiz.map((q, index) => {
                const qsText = marked.parseInline(q.question || '');
                return `
                <div id="qcard-${index}" class="glass-card q-card">
                    <div class="q-label">QUESTION ${index + 1}</div>
                    <div class="q-text md-content">${qsText}</div>
                    <div>
                        ${(q.options || []).map(opt => `
                            <label class="quiz-opt-label">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}">
                                <span class="md-content" style="color: inherit;">${marked.parseInline(opt)}</span>
                            </label>
                        `).join('')}
                    </div>
                    <div id="result-${index}"></div>
                </div>
                `;
            }).join('');
            
            setTimeout(applyMath, 50);
        }

        function checkAnswers() {
            let score = 0;
            AppState.quiz.forEach((q, index) => {
                const selected = document.querySelector(`input[name="question-${index}"]:checked`);
                const resultDiv = document.getElementById(`result-${index}`);
                const card = document.getElementById(`qcard-${index}`);
                
                if (!selected) {
                    resultDiv.innerHTML = `<div style="color: #F43F5E; font-size: 14px; font-weight: 700; margin-top: 12px;"><i class="fa-solid fa-circle-exclamation"></i> Selection required</div>`;
                    return;
                }
                
                const explHtml = marked.parseInline(q.explanation || '');
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box correct md-content">
                            <strong style="color: #34D399; display: block; margin-bottom: 6px; font-size: 15px;">Correct</strong>
                            <span style="color: rgba(255,255,255,0.9);">${explHtml}</span>
                        </div>`;
                    card.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                } else {
                    const corrAns = marked.parseInline(q.correct_answer || '');
                    resultDiv.innerHTML = `
                        <div class="quiz-res-box wrong md-content">
                            <strong style="color: #F43F5E; display: block; margin-bottom: 4px; font-size: 15px;">Incorrect</strong>
                            <div style="font-size: 13px; margin-bottom: 8px;">Correct: <strong style="color: #fff;">${corrAns}</strong></div>
                            <span style="color: rgba(255,255,255,0.9);">${explHtml}</span>
                        </div>`;
                    card.style.borderColor = 'rgba(244, 63, 94, 0.5)';
                }
            });
            
            setTimeout(applyMath, 50);

            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="glass-card" style="text-align: center; margin-top: 28px; padding: 36px 16px; background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05)); border: 1px solid rgba(16, 185, 129, 0.3);">
                        <p style="font-size: 13px; font-weight: 800; color: #34D399; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Final Score</p>
                        <div style="font-size: 48px; font-weight: 800; color: #fff; line-height: 1; text-shadow: 0 4px 16px rgba(16, 185, 129, 0.4);">${score} <span style="font-size: 22px; color: rgba(255,255,255,0.5);">/ ${AppState.quiz.length}</span></div>
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
            chatHistory.innerHTML += `<div class="chat-bubble chat-user md-content">${question}</div>`;
            input.value = '';
            
            chatHistory.scrollTop = chatHistory.scrollHeight;

            const loaderId = 'loader-' + Date.now();
            chatHistory.innerHTML += `
                <div id="${loaderId}" class="chat-bubble chat-ai" style="padding: 16px 18px;">
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
                
                // Parse markdown before injecting
                const formattedAnswer = marked.parse(data.answer || "");
                chatHistory.innerHTML += `<div class="chat-bubble chat-ai md-content">${formattedAnswer}</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
                
                // After adding to DOM, run math renderer
                setTimeout(applyMath, 50);

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