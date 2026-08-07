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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Kaparsh</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        dark: '#000000',
                        card: '#121212',
                        borderline: '#262626',
                        blurple: '#5865F2',
                        famneon: '#00FFA3',
                        xblue: '#1DA1F2',
                        danger: '#ED4245'
                    },
                    fontFamily: {
                        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <style>
        body { background-color: #000; color: #fff; -webkit-tap-highlight-color: transparent; -webkit-user-select: none; user-select: none; }
        ::selection { background: transparent; }
        ::-moz-selection { background: transparent; }
        input { -webkit-user-select: auto; user-select: auto; }
        ::-webkit-scrollbar { display: none; }
        .loader { border-top-color: #00FFA3; animation: spinner 1s linear infinite; }
        @keyframes spinner { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .nav-active { color: #fff !important; }
        .nav-active i { color: #5865F2; }
        .pdf-page { background-color: #000; color: #fff; }
    </style>
</head>
<body class="antialiased overflow-x-hidden pb-36 font-sans">

    <div class="max-w-md mx-auto min-h-screen bg-dark relative border-x border-borderline shadow-2xl">
        
        <header class="sticky top-0 z-40 bg-dark/80 backdrop-blur-md border-b border-borderline px-4 py-3 flex justify-between items-center" data-html2canvas-ignore>
            <h1 class="text-xl font-bold tracking-tight">Kaparsh</h1>
            <button id="download-btn" onclick="downloadNotes()" class="hidden w-8 h-8 rounded-full bg-card border border-borderline flex items-center justify-center active:scale-95 transition-transform">
                <i class="fa-solid fa-arrow-down text-sm"></i>
            </button>
        </header>

        <main class="w-full">
            
            <div id="global-loader" class="hidden flex-col items-center justify-center pt-32 px-6 text-center">
                <div class="loader w-10 h-10 border-4 border-card rounded-full mb-6"></div>
                <p id="loader-title" class="text-base font-bold text-white mb-2">Processing Document...</p>
                <p id="loader-text" class="text-xs font-semibold text-neutral-400">Please wait. AI is analyzing the data.<br>(This may take up to 30 seconds)</p>
            </div>

            <!-- Tab: Home (Micro Analyzer) -->
            <div id="tab-home" class="tab-pane block px-4 py-6">
                <div class="bg-card rounded-3xl p-6 border border-borderline mb-6 relative overflow-hidden">
                    <div class="absolute -right-4 -top-4 w-24 h-24 bg-blurple rounded-full opacity-20 blur-2xl"></div>
                    <h2 class="text-2xl font-bold mb-1">Chapter Analysis</h2>
                    <p class="text-xs text-neutral-400">Upload a single textbook chapter or section for deep-extraction notes, formulas, and quizzes.</p>
                </div>

                <label id="drop-zone" for="file-upload" class="bg-card rounded-3xl p-8 border border-borderline border-dashed flex flex-col items-center justify-center text-center mb-6 active:bg-neutral-900 transition-colors cursor-pointer relative block overflow-hidden">
                    <div class="flex items-center gap-3 mb-3 relative z-10 pointer-events-none">
                        <i class="fa-solid fa-file-pdf text-xl text-blurple"></i>
                        <i class="fa-solid fa-image text-xl text-famneon"></i>
                    </div>
                    <p id="file-name" class="text-sm font-bold text-white relative z-10 pointer-events-none">Upload Chapter (PDF/Img)</p>
                    <input type="file" id="file-upload" accept="application/pdf, image/png, image/jpeg, image/jpg" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
                </label>

                <button id="analyze-btn" class="w-full bg-famneon text-black font-bold text-base py-4 rounded-full active:scale-95 transition-transform shadow-[0_0_15px_rgba(0,255,163,0.3)]">
                    Generate Intelligence
                </button>
            </div>

            <!-- Tab: Notes -->
            <div id="tab-summary" class="tab-pane hidden px-4 py-6 pdf-page">
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-xl font-bold">Notes Feed</h2>
                    <span id="topic-count" class="text-[10px] font-bold bg-card border border-borderline px-3 py-1 rounded-full text-neutral-400 uppercase tracking-wide"></span>
                </div>
                <div id="topics-grid" class="bg-card rounded-3xl border border-borderline p-6 flex flex-col gap-6 shadow-lg">
                    <div class="flex flex-col items-center justify-center py-20 text-center opacity-50">
                        <i class="fa-solid fa-book-open text-4xl mb-4 text-neutral-600"></i>
                        <p class="text-sm font-medium">Your digital notebook is empty.</p>
                    </div>
                </div>
            </div>

            <!-- Tab: Schedule -->
            <div id="tab-schedule" class="tab-pane hidden px-4 py-6">
                <div id="schedule-setup" class="flex flex-col items-center justify-center py-4 text-center">
                    <div class="w-16 h-16 bg-card rounded-full flex items-center justify-center mb-3 border border-borderline">
                        <i class="fa-solid fa-map-location-dot text-2xl text-blurple"></i>
                    </div>
                    <h3 class="text-lg font-bold mb-1">Master Syllabus Planner</h3>
                    <p class="text-xs text-neutral-500 mb-5 px-4">Set your timeline parameters, upload your syllabus, and pick your active subjects.</p>
                    
                    <div class="w-full max-w-sm space-y-3 mb-5">
                        <div class="bg-card rounded-2xl p-4 border border-borderline flex justify-between items-center">
                            <label class="text-sm font-semibold text-neutral-300">Exam Date</label>
                            <input type="date" id="schedule-exam-date" class="bg-transparent text-right text-sm font-bold text-famneon outline-none" style="color-scheme: dark;">
                        </div>
                        <div class="bg-card rounded-2xl p-4 border border-borderline flex justify-between items-center">
                            <label class="text-sm font-semibold text-neutral-300">Daily Hours</label>
                            <input type="number" id="schedule-study-hours" value="2" min="1" max="16" class="bg-transparent text-right text-sm font-bold text-white outline-none w-16">
                        </div>
                    </div>

                    <label id="syllabus-drop-zone" for="syllabus-upload" class="bg-card w-full max-w-sm rounded-3xl p-5 border border-borderline border-dashed flex flex-col items-center justify-center text-center mb-4 active:bg-neutral-900 transition-colors cursor-pointer relative overflow-hidden">
                        <div class="flex items-center gap-3 mb-2 relative z-10 pointer-events-none">
                            <i class="fa-solid fa-file-pdf text-lg text-blurple"></i>
                            <i class="fa-solid fa-image text-lg text-famneon"></i>
                        </div>
                        <p id="syllabus-file-name" class="text-sm font-bold text-white relative z-10 pointer-events-none">Upload Syllabus / Datesheet</p>
                        <input type="file" id="syllabus-upload" accept="application/pdf, image/png, image/jpeg, image/jpg" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
                    </label>

                    <div id="subject-selector-area" class="w-full max-w-sm mb-6 hidden">
                        <p class="text-xs font-bold text-neutral-400 uppercase tracking-wider mb-2 text-left">Select Subjects to Study:</p>
                        <div id="subject-checkboxes" class="bg-card border border-borderline rounded-2xl p-4 flex flex-col gap-3 max-h-48 overflow-y-auto text-left"></div>
                    </div>

                    <button onclick="generateSchedule()" class="w-full max-w-sm bg-blurple text-white font-bold py-4 rounded-full active:scale-95 transition-transform shadow-[0_0_15px_rgba(88,101,242,0.3)]">
                        Build Master Plan
                    </button>
                </div>
                
                <div id="schedule-result" class="hidden flex-col gap-4 relative"></div>
            </div>

            <!-- Tab: Quiz -->
            <div id="tab-quiz" class="tab-pane hidden px-4 py-6">
                <div id="quiz-setup" class="flex flex-col items-center justify-center py-20 text-center">
                    <div class="w-20 h-20 bg-card rounded-full flex items-center justify-center mb-4 border border-borderline">
                        <i class="fa-solid fa-gamepad text-3xl text-famneon"></i>
                    </div>
                    <h3 class="text-lg font-bold mb-2">Dynamic Quiz Engine</h3>
                    <p class="text-sm text-neutral-500 mb-8 px-4">Test your mastery based on the chapter you analyzed.</p>
                    <button onclick="generateQuiz()" class="bg-blurple text-white font-bold py-3 px-8 rounded-full active:scale-95 transition-transform">
                        Start Quiz
                    </button>
                </div>
                <div id="quiz-result" class="hidden flex-col gap-6">
                    <div id="quiz-questions-container" class="flex flex-col gap-6"></div>
                    <button onclick="checkAnswers()" class="w-full bg-famneon text-black font-bold py-4 rounded-full mt-4 active:scale-95 transition-transform">
                        Submit
                    </button>
                    <div id="quiz-score-area"></div>
                </div>
            </div>

            <!-- Tab: Doubts -->
            <div id="tab-doubts" class="tab-pane hidden px-4 py-6 h-full flex flex-col">
                <h2 class="text-xl font-bold mb-4">Doubts</h2>
                <div id="chat-history" class="flex-1 overflow-y-auto flex flex-col gap-4 pb-10">
                    <div class="bg-card p-4 rounded-2xl rounded-tl-sm border border-borderline max-w-[85%] self-start shadow-sm">
                        <p class="text-sm text-neutral-200">Ask me anything about your analyzed chapter!</p>
                    </div>
                </div>
                <div class="fixed bottom-[88px] left-0 right-0 mx-auto w-full max-w-md px-4 z-40">
                    <div class="bg-card border border-borderline rounded-full flex items-center p-1 shadow-lg shadow-black">
                        <input type="text" id="doubt-input" placeholder="Ask a question..." class="flex-1 bg-transparent text-white text-sm px-4 outline-none placeholder-neutral-500">
                        <button onclick="sendDoubt()" class="w-10 h-10 bg-blurple rounded-full flex items-center justify-center text-white active:scale-95 transition-transform shrink-0">
                            <i class="fa-solid fa-arrow-up"></i>
                        </button>
                    </div>
                </div>
            </div>
        </main>

        <nav class="fixed bottom-0 w-full max-w-md bg-dark/95 backdrop-blur-xl border-t border-borderline flex justify-around items-center pb-6 pt-3 z-50">
            <button class="nav-btn active text-neutral-500 flex flex-col items-center gap-1 w-14" data-target="tab-home">
                <i class="fa-solid fa-house text-[22px]"></i>
                <span class="text-[10px] font-medium">Home</span>
            </button>
            <button class="nav-btn text-neutral-500 flex flex-col items-center gap-1 w-14" data-target="tab-summary">
                <i class="fa-solid fa-layer-group text-[22px]"></i>
                <span class="text-[10px] font-medium">Notes</span>
            </button>
            <button class="nav-btn text-neutral-500 flex flex-col items-center gap-1 w-14" data-target="tab-schedule">
                <i class="fa-solid fa-calendar-day text-[22px]"></i>
                <span class="text-[10px] font-medium">Plan</span>
            </button>
            <button class="nav-btn text-neutral-500 flex flex-col items-center gap-1 w-14" data-target="tab-quiz">
                <i class="fa-solid fa-gamepad text-[22px]"></i>
                <span class="text-[10px] font-medium">Quiz</span>
            </button>
            <button class="nav-btn text-neutral-500 flex flex-col items-center gap-1 w-14" data-target="tab-doubts">
                <i class="fa-solid fa-comment-dots text-[22px]"></i>
                <span class="text-[10px] font-medium">Doubts</span>
            </button>
        </nav>

    </div>

    <script>
        const AppState = {
            extractedText: "",
            topics: null,
            schedule: null,
            quiz: null,
            file: null,
            syllabusFile: null,
            syllabusContextText: "",
            globalBannedTerms: [] 
        };

        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 14);
        document.getElementById('schedule-exam-date').valueAsDate = tomorrow;

        const fileInput = document.getElementById('file-upload');
        const fileNameDisplay = document.getElementById('file-name');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                AppState.file = fileInput.files[0];
                fileNameDisplay.textContent = AppState.file.name;
                fileNameDisplay.classList.add('text-famneon');
            }
        });
        
        const syllabusInput = document.getElementById('syllabus-upload');
        const syllabusNameDisplay = document.getElementById('syllabus-file-name');
        
        syllabusInput.addEventListener('change', async () => {
            if (syllabusInput.files.length > 0) {
                AppState.syllabusFile = syllabusInput.files[0];
                syllabusNameDisplay.textContent = AppState.syllabusFile.name;
                syllabusNameDisplay.classList.add('text-blurple');
                
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
                <label class="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" name="syllabus-subject" value="${sub}" checked class="w-4 h-4 accent-blurple rounded bg-dark border-borderline">
                    <span class="text-xs text-neutral-200 font-medium">${sub}</span>
                </label>
            `).join('');
            
            area.classList.remove('hidden');
        }

        const navBtns = document.querySelectorAll('.nav-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');

        function switchTab(targetId) {
            navBtns.forEach(btn => btn.classList.remove('nav-active'));
            document.querySelector(`.nav-btn[data-target="${targetId}"]`).classList.add('nav-active');
            tabPanes.forEach(pane => pane.classList.add('hidden'));
            document.getElementById(targetId).classList.remove('hidden');
            window.scrollTo(0, 0);
        }

        navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.dataset.target;
                if (target === 'tab-summary' || target === 'tab-quiz' || target === 'tab-doubts') {
                    if (!AppState.topics) return alert("Upload and analyze a Chapter first!");
                }
                switchTab(target);
            });
        });

        const toggleLoader = (show, title = 'Processing Document...', text = 'Please wait. AI is analyzing the data.<br>(This may take up to 30 seconds)') => {
            document.getElementById('loader-title').innerText = title;
            document.getElementById('loader-text').innerHTML = text;
            
            if (show) {
                tabPanes.forEach(pane => pane.classList.add('hidden'));
                document.getElementById('global-loader').classList.remove('hidden');
                document.getElementById('global-loader').classList.add('flex');
            } else {
                document.getElementById('global-loader').classList.add('hidden');
                document.getElementById('global-loader').classList.remove('flex');
            }
        };

        const showError = (msg) => {
            alert("Error: " + msg);
            toggleLoader(false);
            switchTab('tab-home');
        };

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
                switchTab('tab-summary');
                toggleLoader(false);
                
                for (let i = 0; i < AppState.topics.length; i++) {
                    await fetchTopicDetails(i);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

                AppState.quiz = null;
                document.getElementById('quiz-result').classList.add('hidden');
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
                    <div class="border-b border-borderline pb-6 mb-6 opacity-50 animate-pulse">
                        <div class="h-5 bg-borderline rounded w-2/3 mb-3"></div>
                        <div class="h-3 bg-borderline rounded w-3/4 mb-2"></div>
                    </div>`;
                }

                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div class="mt-4 space-y-2">
                        ${t.definitions.map(d => `
                            <div class="bg-blurple/10 border-l-2 border-blurple p-3 rounded-r-xl">
                                <p class="text-sm text-white"><strong class="text-blurple font-bold mr-2">${d.term}:</strong>${d.definition}</p>
                            </div>
                        `).join('')}
                    </div>` : '';

                const formulasHtml = (t.formulas && t.formulas.length > 0) ? `
                    <div class="mt-4 space-y-2">
                        ${t.formulas.map(f => `
                            <div class="bg-famneon/5 border border-famneon/20 p-4 rounded-xl flex flex-col items-center">
                                <p class="font-mono text-lg font-bold text-famneon">${f.equation}</p>
                                <p class="text-xs text-white font-medium mt-1">${f.meaning}</p>
                            </div>
                        `).join('')}
                    </div>` : '';

                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div class="mt-4 space-y-2">
                        ${t.derivations.map(d => `
                            <div class="bg-dark border border-xblue/30 p-4 rounded-xl">
                                <p class="text-sm font-bold text-xblue mb-1">${d.title}</p>
                                <p class="font-mono text-xs text-white leading-relaxed whitespace-pre-wrap">${d.content}</p>
                            </div>
                        `).join('')}
                    </div>` : '';

                const notesList = (t.notes || []).map(n => `<li class="mb-2 flex items-start text-sm text-white"><span class="text-famneon mr-2 mt-1 text-[10px]"><i class="fa-solid fa-circle"></i></span>${n}</li>`).join('');

                return `
                <div class="border-b border-borderline pb-6 mb-2 last:border-0 last:mb-0 last:pb-0">
                    <div class="flex justify-between items-start mb-3">
                        <h4 class="font-bold text-lg text-white leading-tight pr-3">${t.title}</h4>
                        <span class="text-[10px] font-bold px-2 py-1 rounded bg-dark border border-borderline text-neutral-400 uppercase tracking-widest">${t.priority || 'MED'}</span>
                    </div>
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    <ul class="mt-4 space-y-1">${notesList}</ul>
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

        // Macro-Planner Logic
        async function generateSchedule() {
            const examDate = document.getElementById('schedule-exam-date').value;
            const studyHours = document.getElementById('schedule-study-hours').value;
            
            if (!examDate) {
                return alert("Please set your Exam Date in the Plan tab first.");
            }

            const selectedSubjects = [];
            document.querySelectorAll('input[name="syllabus-subject"]:checked').forEach(cb => {
                selectedSubjects.push(cb.value);
            });

            toggleLoader(true, 'Building Master Plan...', 'Mapping out your timeline based on your selected subjects...');

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
                } 
                else if (AppState.topics && AppState.topics.length > 0) {
                    const loadedTopics = AppState.topics.filter(t => t.loaded);
                    response = await fetch('/api/schedule', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ topics: loadedTopics, exam_date: examDate, study_hours: parseFloat(studyHours) })
                    });
                } else {
                    throw new Error("Please upload a Master Syllabus OR analyze a Chapter on the Home tab first.");
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
            const container = document.getElementById('schedule-result');
            if (!AppState.schedule || !Array.isArray(AppState.schedule)) {
                return alert("Invalid schedule data received.");
            }
            
            container.innerHTML = AppState.schedule.map((day) => `
                <div class="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-borderline before:to-transparent">
                    <div class="flex items-center justify-center w-10 h-10 rounded-full border border-white bg-card text-white shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                        <span class="text-xs font-bold">${day.day || '-'}</span>
                    </div>
                    <div class="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] bg-card p-5 rounded-2xl border border-borderline z-10">
                        <div class="flex justify-between items-start mb-1">
                            <span class="text-famneon text-xs font-bold">${(day.date || '').split('-').slice(1).join('/')}</span>
                            <span class="text-neutral-500 text-[10px] font-bold"><i class="fa-regular fa-clock"></i> ${day.total_hours_today || 0}h</span>
                        </div>
                        <h4 class="font-bold text-white text-sm mb-3">${day.focus_area || 'Study Block'}</h4>
                        <div class="flex flex-col gap-2 mb-4">
                            ${(day.topics || []).map(t => `
                                <div class="flex justify-between items-center bg-dark p-2 rounded-xl border border-borderline">
                                    <span class="text-xs text-neutral-300 font-medium truncate pr-2">${t.name || 'Topic'}</span>
                                    <span class="text-[10px] text-blurple font-bold whitespace-nowrap bg-blurple/10 px-2 py-1 rounded">${t.estimated_minutes || 30} min</span>
                                </div>
                            `).join('')}
                        </div>
                        <p class="text-xs text-neutral-400 leading-relaxed border-l-2 border-borderline pl-2">${day.actionable_advice || ''}</p>
                    </div>
                </div>
            `).join('');
        }

        async function generateQuiz() {
            toggleLoader(true, 'Generating Exam...', 'Building difficult multiple-choice questions based on your material.');
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
                <div id="qcard-${index}" class="bg-card p-6 rounded-3xl border border-borderline">
                    <p class="text-blurple font-bold text-xs mb-2">QUESTION ${index + 1}</p>
                    <h4 class="font-bold text-base mb-5 text-white leading-relaxed">${q.question || 'Missing question'}</h4>
                    <div class="space-y-3">
                        ${(q.options || []).map(opt => `
                            <label class="flex items-center gap-3 cursor-pointer p-4 rounded-2xl border border-borderline bg-dark active:bg-neutral-900 transition-colors">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}" class="w-5 h-5 accent-blurple bg-dark border-borderline">
                                <span class="text-sm text-neutral-200 font-medium">${opt}</span>
                            </label>
                        `).join('')}
                    </div>
                    <div id="result-${index}" class="mt-4"></div>
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
                    resultDiv.innerHTML = `<div class="text-sm text-danger font-bold py-2"><i class="fa-solid fa-circle-exclamation mr-1"></i> Answer required</div>`;
                    return;
                }
                
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="p-4 bg-famneon/10 border border-famneon/30 rounded-2xl mt-4">
                            <p class="text-famneon font-bold text-sm mb-1">Correct</p>
                            <p class="text-xs text-famneon/80 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-famneon/50');
                    card.classList.remove('border-borderline');
                } else {
                    resultDiv.innerHTML = `
                        <div class="p-4 bg-danger/10 border border-danger/30 rounded-2xl mt-4">
                            <p class="text-danger font-bold text-sm mb-2">Incorrect</p>
                            <p class="text-xs mb-3 text-white">Correct: <span class="font-bold text-danger">${q.correct_answer}</span></p>
                            <p class="text-xs text-danger/80 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-danger/50');
                    card.classList.remove('border-borderline');
                }
            });
            
            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="mt-6 p-8 bg-card rounded-3xl text-center border border-borderline">
                        <p class="text-xs font-bold uppercase tracking-widest text-neutral-500 mb-2">Final Score</p>
                        <h3 class="text-5xl font-black text-white mb-2">${score} <span class="text-neutral-600 text-3xl">/ ${AppState.quiz.length}</span></h3>
                    </div>
                `;
            }
        }

        document.getElementById('doubt-input').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendDoubt();
        });

        async function sendDoubt() {
            const input = document.getElementById('doubt-input');
            const question = input.value.trim();
            if (!question) return;
            if (!AppState.extractedText) return alert("Please upload and process a Chapter first to ask doubts.");

            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML += `<div class="bg-blurple text-white p-4 rounded-2xl rounded-tr-sm max-w-[85%] self-end shadow-md"><p class="text-sm">${question}</p></div>`;
            input.value = '';
            window.scrollTo(0, document.body.scrollHeight);

            const loaderId = 'loader-' + Date.now();
            chatHistory.innerHTML += `
                <div id="${loaderId}" class="bg-card p-4 rounded-2xl rounded-tl-sm border border-borderline max-w-[85%] self-start flex items-center gap-1.5 shadow-sm">
                    <div class="w-2 h-2 bg-neutral-500 rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-neutral-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <div class="w-2 h-2 bg-neutral-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                </div>`;
            window.scrollTo(0, document.body.scrollHeight);

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
                chatHistory.innerHTML += `<div class="bg-card p-4 rounded-2xl rounded-tl-sm border border-borderline max-w-[90%] self-start shadow-sm"><p class="text-sm text-neutral-200 leading-relaxed">${formattedAnswer}</p></div>`;
                window.scrollTo(0, document.body.scrollHeight);
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