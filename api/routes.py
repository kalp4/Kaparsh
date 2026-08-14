import json
from flask import Blueprint, request, jsonify
from google.genai import types
from services.gemini_service import get_gemini_client, upload_and_wait_active

api_bp = Blueprint('api', __name__)

@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Kaparsh Flask Backend is Live and Ready!"})

@api_bp.route("/analyze", methods=["POST"])
def analyze_pdf():
    try:
        client = get_gemini_client()
        
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded."}), 400
            
        file = request.files['file']
        filename = file.filename.lower()
        
        if not (filename.endswith(".pdf") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            return jsonify({"detail": "Only PDF and Image files (.pdf, .png, .jpg, .jpeg) are supported."}), 400
            
        gemini_file_name = upload_and_wait_active(client, file)
        gemini_file = client.files.get(name=gemini_file_name)
            
        prompt = (
            "You are an AI study assistant. Read the attached document and identify all distinct, primary concepts.\n"
            "Ensure there are NO duplicate or heavily overlapping topics. Merge similar concepts into a single title.\n"
            "Return ONLY a JSON object with this exact structure (no extra markdown):\n"
            "{\n"
            '  "topics": [\n'
            '    {"title": "Exact Topic Name"}\n'
            "  ]\n"
            "}\n"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=[prompt, gemini_file],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        parsed_data = json.loads(response.text)
        
        return jsonify({
            "topics": parsed_data.get("topics", []),
            "gemini_file_name": gemini_file_name
        })
        
    except Exception as e:
        return jsonify({"detail": f"File Analysis failed: {str(e)}"}), 500

@api_bp.route("/parse-syllabus", methods=["POST"])
def parse_syllabus():
    try:
        client = get_gemini_client()
        
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded."}), 400
            
        file = request.files['file']
        filename = file.filename.lower()
        
        if not (filename.endswith(".pdf") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            return jsonify({"detail": "Only PDF and Image files (.pdf, .png, .jpg, .jpeg) are supported."}), 400
            
        gemini_file_name = upload_and_wait_active(client, file)
        gemini_file = client.files.get(name=gemini_file_name)
            
        prompt = (
            "Analyze the attached syllabus document and extract a clean list of all distinct subjects or modules found in it.\n"
            "Return ONLY a JSON object with this exact structure (no extra markdown):\n"
            "{\n"
            '  "subjects": ["Subject 1", "Subject 2", "Subject 3"]\n'
            "}\n"
        )
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=[prompt, gemini_file],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        parsed = json.loads(response.text)
        subjects = parsed.get("subjects", [])
        
        if not subjects:
            subjects = ["General Coursework / Core Syllabus"]

        return jsonify({
            "subjects": subjects,
            "gemini_file_name": gemini_file_name
        })
        
    except Exception as e:
        return jsonify({"detail": f"Syllabus parsing failed: {str(e)}"}), 500

@api_bp.route("/topic", methods=["POST"])
def get_topic_details():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        topic_name = data.get('topic')
        covered_topics = data.get('covered_topics', '')
        gemini_file_name = data.get('gemini_file_name', '')
        
        ignore_prompt = f"GLOBAL BAN LIST: The following concepts have ALREADY been defined: [{covered_topics}]. DO NOT DEFINE THEM AGAIN.\n" if covered_topics else ""
        
        prompt = (
            f"Extract detailed study materials EXCLUSIVELY for the specific topic: '{topic_name}' from the attached document.\n"
            f"{ignore_prompt}"
            "Categorize information strictly into definitions, formulas, derivations, and general notes.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. MUTUALLY EXCLUSIVE: If a fact is a 'definition', do not repeat it in 'notes'.\n"
            "2. BE HIGHLY CONCISE: Strip out filler words.\n"
            "3. MATH RENDERING: ALL mathematical symbols, variables, expressions, and equations MUST be wrapped in KaTeX delimiters. Use single $ for inline math (e.g., $E=mc^2$). NEVER place single $ on their own lines for multiline equations.\n"
            "4. DERIVATIONS: Format derivations as step-by-step lists. EVERY major mathematical step or equation in a derivation MUST be formatted as an inline equation (e.g. $I = \\frac{e}{T}$) or wrapped in double dollar signs on a single line (e.g., $$ I = \\frac{e}{T} $$). Do NOT place single $ on separate lines above and below equations.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "priority": "High",\n'
            '  "definitions": [{"term": "Exact Term", "definition": "Clear definition"}],\n'
            '  "formulas": [{"equation": "E=mc^2", "meaning": "Meaning"}],\n'
            '  "derivations": [{"title": "Derivation Name", "content": "1. Step one: $I = \\\\frac{e}{T}$\\n\\n2. Step two: $I = \\\\frac{ev}{2\\\\pi r}$"}],\n'
            '  "notes": ["Concise point 1", "Concise point 2"]\n'
            "}\n"
        )
        
        contents = [prompt]
        if gemini_file_name:
            contents.append(client.files.get(name=gemini_file_name))
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Topic detailing failed: {str(e)}"}), 500

@api_bp.route("/schedule", methods=["POST"])
def generate_schedule():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        exam_date = data.get('exam_date', '')
        study_hours = data.get('study_hours', 2)
        selected_subjects = data.get('selected_subjects', [])
        syllabus_text = data.get('syllabus_text', '')
        gemini_file_name = data.get('gemini_file_name', '')
        topics = data.get('topics', [])
        
        contents = []
        
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
            contents.append(prompt)
        else:
            subjects_str = ", ".join(selected_subjects) if selected_subjects else "All Selected Subjects"
            prompt = (
                f"You are a Master Study Planner. Build a custom study schedule based on the attached syllabus document.\n"
                f"Target Exam Date: {exam_date}\n"
                f"Daily Study Hours Available: {study_hours}\n"
                f"Strictly limit the schedule ONLY to these selected subjects: [{subjects_str}]. Completely ignore any other subjects or modules mentioned in the syllabus text.\n\n"
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
            contents.append(prompt)
            if gemini_file_name:
                contents.append(client.files.get(name=gemini_file_name))
            elif syllabus_text:
                contents.append(f"Syllabus Content:\n{syllabus_text[:30000]}")
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Schedule generation failed: {str(e)}"}), 500

@api_bp.route("/quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        client = get_gemini_client()
        topics_json = json.dumps(data.get('topics', []))
        
        prompt = (
            "Create a multiple-choice practice exam based on these topics.\n"
            "CRITICAL INSTRUCTION: Any mathematical symbols, variables, equations, or formulas used in the question, options, or explanation MUST be wrapped in $ (e.g., $E=mc^2$) for KaTeX rendering. For standalone block equations, use $$.\n"
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
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return jsonify(json.loads(response.text))
        
    except Exception as e:
        return jsonify({"detail": f"Quiz generation failed: {str(e)}"}), 500

@api_bp.route("/doubt", methods=["POST"])
def answer_doubt():
    try:
        data = request.get_json()
        client = get_gemini_client()
        
        question = data.get('question')
        gemini_file_name = data.get('gemini_file_name')
        
        prompt = (
            "You are a helpful expert tutor. A student has a doubt regarding their study material.\n\n"
            f"Student's Doubt: {question}\n\n"
            "Provide a clear, concise, and accurate explanation based ONLY on the context provided in the attached document."
        )
        
        contents = [prompt]
        if gemini_file_name:
            contents.append(client.files.get(name=gemini_file_name))
        else:
            contents.append("No context document was uploaded. Answer the doubt accurately using general educational knowledge.")
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=contents
        )
        
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"detail": f"Failed to answer doubt: {str(e)}"}), 500