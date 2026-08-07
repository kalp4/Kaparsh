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


# ==============================================================================
# FRONTEND INJECTION
# ==============================================================================

KAPARSH_FRONTEND = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaparsh — AI Study & Revision Agent</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#eef2ff',
                            100: '#e0e7ff',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca',
                            900: '#312e81',
                        }
                    },
                    boxShadow: {
                        'soft': '0 4px 20px -2px rgba(0, 0, 0, 0.05)',
                        'float': '0 10px 30px -5px rgba(99, 102, 241, 0.15)',
                    }
                }
            }
        }
    </script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- html2pdf for Native Download -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <style>
        body {
            background-color: #f8fafc;
        }
        .loader {
            border-top-color: #4f46e5;
            -webkit-animation: spinner 1s linear infinite;
            animation: spinner 1s linear infinite;
        }
        @keyframes spinner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .drag-active {
            border-color: #4f46e5 !important;
            background-color: #eef2ff !important;
        }
        .tab-active {
            border-bottom: 3px solid #4f46e5;
            color: #4f46e5;
            font-weight: 600;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        
        /* A4 Page Styling for the PDF Export */
        .pdf-page {
            background: white;
            padding: 1rem;
        }
    </style>
</head>
<body class="min-h-screen font-sans antialiased text-slate-800 selection:bg-brand-100 selection:text-brand-900">

    <!-- Header -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white shadow-md shadow-brand-500/30">
                    <i class="fa-solid fa-graduation-cap text-sm"></i>
                </div>
                <h1 class="text-xl font-bold tracking-tight text-slate-900">Kaparsh</h1>
            </div>
            <div class="text-sm font-medium text-slate-500 hidden sm:block">AI Study & Revision Agent</div>
        </div>
    </header>

    <!-- Main Layout -->
    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Sidebar -->
        <aside class="lg:col-span-4 flex flex-col gap-6" data-html2canvas-ignore>
            <div class="bg-white rounded-2xl border border-slate-200 p-6 shadow-soft">
                <h2 class="text-base font-bold mb-5 text-slate-900 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-brand-500"></i> Configuration
                </h2>
                
                <div class="space-y-5">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-500 mb-1.5 uppercase tracking-wide">Exam Date</label>
                            <input type="date" id="exam-date" class="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none transition-all bg-slate-50">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-500 mb-1.5 uppercase tracking-wide">Hours / Day</label>
                            <input type="number" id="study-hours" value="2" min="1" max="16" class="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none transition-all bg-slate-50">
                        </div>
                    </div>
                </div>

                <div class="mt-6 pt-6 border-t border-slate-100">
                    <label class="block text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wide">Upload Material</label>
                    <div id="drop-zone" class="border-2 border-dashed border-slate-300 rounded-xl bg-slate-50/50 p-8 text-center hover:bg-brand-50 hover:border-brand-300 transition-all cursor-pointer group">
                        <div class="w-12 h-12 bg-white rounded-full flex items-center justify-center mx-auto mb-3 shadow-sm group-hover:scale-110 transition-transform duration-300">
                            <i class="fa-regular fa-file-pdf text-xl text-brand-500"></i>
                        </div>
                        <p class="text-sm text-slate-900 font-bold mb-1">Select Textbook PDF</p>
                        <p id="file-name" class="text-xs text-slate-500 font-medium">Auto-scales up to 25 pages</p>
                        <input type="file" id="file-upload" accept=".pdf" class="hidden">
                    </div>
                </div>

                <button id="analyze-btn" class="w-full mt-6 bg-brand-600 hover:bg-brand-700 text-white text-sm font-bold py-3.5 px-4 rounded-xl shadow-md shadow-brand-500/25 hover:shadow-float hover:-translate-y-0.5 transition-all duration-300 flex items-center justify-center gap-2">
                    Process Document <i class="fa-solid fa-arrow-right"></i>
                </button>
            </div>
        </aside>

        <!-- Main Content Area -->
        <section class="lg:col-span-8">
            <div class="bg-white rounded-2xl border border-slate-200 min-h-[600px] flex flex-col shadow-soft overflow-hidden">
                
                <!-- Tabs -->
                <div class="flex border-b border-slate-200 bg-slate-50/50" data-html2canvas-ignore>
                    <button class="tab-btn flex-1 py-4 text-sm font-medium text-slate-500 hover:text-brand-600 transition-colors tab-active flex items-center justify-center gap-2" data-target="tab-summary">
                        <i class="fa-solid fa-book-open"></i> Study Notes
                    </button>
                    <button class="tab-btn flex-1 py-4 text-sm font-medium text-slate-500 hover:text-brand-600 transition-colors flex items-center justify-center gap-2" data-target="tab-schedule">
                        <i class="fa-solid fa-calendar-day"></i> Schedule
                    </button>
                    <button class="tab-btn flex-1 py-4 text-sm font-medium text-slate-500 hover:text-brand-600 transition-colors flex items-center justify-center gap-2" data-target="tab-quiz">
                        <i class="fa-solid fa-flask"></i> Practice Exam
                    </button>
                </div>

                <!-- Loader -->
                <div id="global-loader" class="hidden flex-1 flex flex-col items-center justify-center p-12">
                    <div class="loader w-10 h-10 border-4 border-slate-100 rounded-full mb-5"></div>
                    <p id="loader-text" class="text-sm font-bold text-brand-600 animate-pulse tracking-wide">Mapping Cognitive Structures...</p>
                </div>

                <!-- Empty State -->
                <div id="empty-state" class="flex-1 flex flex-col items-center justify-center p-12 text-center">
                    <div class="w-16 h-16 bg-brand-50 flex items-center justify-center mb-5 rounded-2xl shadow-inner border border-brand-100">
                        <i class="fa-solid fa-layer-group text-2xl text-brand-500"></i>
                    </div>
                    <h3 class="text-lg font-bold text-slate-900 mb-2">Workspace Ready</h3>
                    <p class="text-sm text-slate-500 max-w-sm leading-relaxed">Upload a chapter to generate perfectly color-coded notes, formulas, and derivations.</p>
                </div>

                <!-- Tab Contents -->
                <div id="tabs-container" class="hidden flex-1 p-8 overflow-y-auto bg-slate-50/30">
                    
                    <!-- Notes Tab -->
                    <div id="tab-summary" class="tab-pane block pdf-page">
                        <div class="flex justify-between items-center mb-8 pb-4 border-b border-slate-200">
                            <div class="flex items-center gap-4">
                                <h2 class="text-2xl font-bold text-slate-900 tracking-tight">Structured Notes</h2>
                                <span id="topic-count" class="text-xs font-bold text-brand-600 bg-brand-50 border border-brand-100 px-2.5 py-1 rounded-md"></span>
                            </div>
                            
                            <!-- Save as PDF Button -->
                            <button id="download-btn" onclick="downloadNotes()" class="hidden px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow-md transition-all flex items-center gap-2" data-html2canvas-ignore>
                                <i class="fa-solid fa-download"></i> Download PDF
                            </button>
                        </div>
                        <div id="topics-grid" class="flex flex-col gap-8">
                            <!-- Injected by JS -->
                        </div>
                    </div>

                    <!-- Schedule Tab -->
                    <div id="tab-schedule" class="tab-pane hidden">
                        <div id="schedule-setup" class="text-center py-20">
                            <div class="w-16 h-16 bg-brand-50 mx-auto rounded-2xl flex items-center justify-center mb-5 text-brand-500 text-2xl">
                                <i class="fa-regular fa-calendar-check"></i>
                            </div>
                            <h3 class="text-xl font-bold text-slate-900 mb-2">Generate Study Timeline</h3>
                            <p class="text-sm text-slate-500 mb-8 max-w-sm mx-auto">Map your extracted topics to an algorithmic spaced repetition schedule.</p>
                            <button onclick="generateSchedule()" class="px-6 py-3 bg-brand-600 hover:bg-brand-700 text-white text-sm font-bold rounded-xl shadow-md shadow-brand-500/25 hover:-translate-y-0.5 transition-all">
                                Generate Plan
                            </button>
                        </div>
                        <div id="schedule-result" class="hidden space-y-4">
                            <!-- Injected by JS -->
                        </div>
                    </div>

                    <!-- Quiz Tab -->
                    <div id="tab-quiz" class="tab-pane hidden">
                        <div id="quiz-setup" class="text-center py-20">
                            <div class="w-16 h-16 bg-brand-50 mx-auto rounded-2xl flex items-center justify-center mb-5 text-brand-500 text-2xl">
                                <i class="fa-solid fa-spell-check"></i>
                            </div>
                            <h3 class="text-xl font-bold text-slate-900 mb-2">Knowledge Verification</h3>
                            <p class="text-sm text-slate-500 mb-8 max-w-sm mx-auto">Test your comprehension with an AI-generated exam built from your document.</p>
                            <button onclick="generateQuiz()" class="px-6 py-3 bg-brand-600 hover:bg-brand-700 text-white text-sm font-bold rounded-xl shadow-md shadow-brand-500/25 hover:-translate-y-0.5 transition-all">
                                Start Exam
                            </button>
                        </div>
                        <div id="quiz-result" class="hidden space-y-8">
                            <div id="quiz-questions-container" class="space-y-8"></div>
                            <button onclick="checkAnswers()" class="w-full py-4 bg-slate-900 text-white text-base font-bold rounded-xl mt-4 hover:bg-slate-800 transition-colors shadow-md hover:shadow-lg">
                                Submit Answers
                            </button>
                            <div id="quiz-score-area"></div>
                        </div>
                    </div>

                </div>
            </div>
        </section>
    </main>

    <script>
        const AppState = {
            extractedText: "",
            topics: null,
            schedule: null,
            quiz: null,
            file: null
        };

        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 7);
        document.getElementById('exam-date').valueAsDate = tomorrow;

        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-upload');
        const fileNameDisplay = document.getElementById('file-name');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('bg-slate-100'); });
        dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('bg-slate-100'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('bg-slate-100');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            if (fileInput.files.length > 0) {
                AppState.file = fileInput.files[0];
                fileNameDisplay.textContent = AppState.file.name;
                fileNameDisplay.classList.add('text-brand-600', 'font-bold');
            }
        }

        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                if (!AppState.topics) return; 
                tabBtns.forEach(b => b.classList.remove('tab-active'));
                btn.classList.add('tab-active');
                tabPanes.forEach(pane => pane.classList.add('hidden'));
                document.getElementById(btn.dataset.target).classList.remove('hidden');
            });
        });

        const toggleLoader = (show, text = 'Processing...') => {
            document.getElementById('loader-text').innerText = text;
            if (show) {
                document.getElementById('empty-state').classList.add('hidden');
                document.getElementById('tabs-container').classList.add('hidden');
                document.getElementById('global-loader').classList.remove('hidden');
            } else {
                document.getElementById('global-loader').classList.add('hidden');
                document.getElementById('tabs-container').classList.remove('hidden');
            }
        };

        const showError = (msg) => {
            alert("Error: " + msg);
            toggleLoader(false);
            if (!AppState.topics) {
                document.getElementById('tabs-container').classList.add('hidden');
                document.getElementById('empty-state').classList.remove('hidden');
            }
        };

        document.getElementById('analyze-btn').addEventListener('click', async () => {
            if (!AppState.file) return alert("Please upload a PDF file.");

            toggleLoader(true, 'Extracting conceptual framework...');

            const formData = new FormData();
            formData.append('file', AppState.file);

            try {
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                const rawText = await response.text();
                
                if (!response.ok) {
                    try {
                        const errJson = JSON.parse(rawText);
                        throw new Error(errJson.detail || "Server Error");
                    } catch (parseErr) {
                        throw new Error(`Network Error (${response.status}): The backend timed out or could not be reached.`);
                    }
                }
                
                const data = JSON.parse(rawText);
                AppState.extractedText = data.extracted_text;
                
                AppState.topics = data.topics.map(t => ({ title: t.title, loaded: false }));
                
                renderTopics();
                tabBtns[0].click();
                toggleLoader(false);
                
                for (let i = 0; i < AppState.topics.length; i++) {
                    await fetchTopicDetails(i);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

                AppState.schedule = null;
                AppState.quiz = null;
                document.getElementById('schedule-result').classList.add('hidden');
                document.getElementById('schedule-setup').classList.remove('hidden');
                document.getElementById('quiz-result').classList.add('hidden');
                document.getElementById('quiz-setup').classList.remove('hidden');

            } catch (err) {
                showError(err.message);
            }
        });

        async function fetchTopicDetails(index) {
            const topic = AppState.topics[index];
            try {
                const response = await fetch('/api/topic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: AppState.extractedText,
                        topic: topic.title
                    })
                });
                
                if (response.ok) {
                    const details = await response.json();
                    AppState.topics[index] = { ...topic, ...details, loaded: true };
                    renderTopics(); 
                }
            } catch (e) {
                console.error("Failed to load details for", topic.title);
            }
        }

        function renderTopics() {
            const grid = document.getElementById('topics-grid');
            document.getElementById('topic-count').innerText = `${AppState.topics.length} Core Topics`;
            
            const downloadBtn = document.getElementById('download-btn');
            if (AppState.topics.some(t => t.loaded)) {
                downloadBtn.classList.remove('hidden');
            }
            
            grid.innerHTML = AppState.topics.map((t) => {
                if (!t.loaded) {
                    return `
                    <div class="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm opacity-70 animate-pulse">
                        <div class="flex justify-between items-center mb-5">
                            <h4 class="font-bold text-lg text-slate-800">${t.title}</h4>
                            <div class="w-5 h-5 border-2 border-slate-200 border-t-brand-500 rounded-full animate-spin"></div>
                        </div>
                        <div class="space-y-3">
                            <div class="h-2 bg-slate-100 rounded-full w-3/4"></div>
                            <div class="h-2 bg-slate-100 rounded-full w-1/2"></div>
                            <div class="h-2 bg-slate-100 rounded-full w-5/6"></div>
                        </div>
                    </div>`;
                }

                const priorityStyles = {
                    'High': 'bg-red-50 text-red-600 border-red-200',
                    'Medium': 'bg-slate-100 text-slate-600 border-slate-200',
                    'Low': 'bg-emerald-50 text-emerald-600 border-emerald-200'
                };
                const pStyle = priorityStyles[t.priority] || priorityStyles['Medium'];
                
                // 1. Vibrant Definitions
                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div class="my-5 space-y-3">
                        ${t.definitions.map(d => `
                            <p class="text-sm text-slate-700 leading-relaxed bg-blue-50/50 p-3 rounded-xl border border-blue-100/50">
                                <strong class="text-blue-600 font-bold text-base mr-1 drop-shadow-sm">${d.term}:</strong> ${d.definition}
                            </p>
                        `).join('')}
                    </div>
                ` : '';

                // 2. Boxed & Highlighted Formulas
                const formulasHtml = (t.formulas && t.formulas.length > 0) ? `
                    <div class="my-6 space-y-3">
                        ${t.formulas.map(f => `
                            <div class="bg-amber-50 border border-amber-300 p-5 rounded-xl shadow-inner flex flex-col items-center gap-1 break-inside-avoid relative overflow-hidden">
                                <div class="absolute left-0 top-0 bottom-0 w-1.5 bg-amber-400"></div>
                                <p class="font-mono text-xl font-bold text-amber-900 tracking-wide">${f.equation}</p>
                                <p class="text-xs text-amber-700 uppercase tracking-widest font-bold mt-1">${f.meaning}</p>
                            </div>
                        `).join('')}
                    </div>
                ` : '';

                // 3. Different Color Derivations
                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div class="my-6 space-y-3">
                        ${t.derivations.map(d => `
                            <div class="bg-emerald-50/50 border-l-4 border-emerald-500 p-5 rounded-r-xl shadow-sm break-inside-avoid">
                                <p class="text-sm font-bold text-emerald-800 uppercase tracking-wider mb-2 flex items-center gap-2"><i class="fa-solid fa-code-branch"></i> ${d.title}</p>
                                <p class="text-sm text-emerald-900 font-mono leading-relaxed whitespace-pre-wrap pl-2 border-l border-emerald-200">${d.content}</p>
                            </div>
                        `).join('')}
                    </div>
                ` : '';

                // General Notes
                const notesList = t.notes.map(n => `<li class="mb-2.5 flex items-start text-sm text-slate-700 leading-relaxed"><div class="mt-1.5 mr-3 w-1.5 h-1.5 rounded-full bg-slate-300 flex-shrink-0"></div>${n}</li>`).join('');

                return `
                <div class="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm hover:shadow-md transition-shadow duration-300 break-inside-avoid">
                    <div class="flex justify-between items-start mb-6">
                        <h4 class="font-bold text-xl text-slate-900 tracking-tight pr-4">${t.title}</h4>
                        <span class="text-[10px] font-bold px-2.5 py-1 rounded-md border uppercase tracking-wider ${pStyle}">${t.priority}</span>
                    </div>
                    
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    
                    <ul class="mb-6 mt-6">
                        ${notesList}
                    </ul>
                    
                    <div class="mt-6 pt-5 border-t border-slate-100 bg-slate-50 rounded-xl p-5 border border-slate-100/50">
                        <p class="text-[10px] uppercase font-bold text-brand-500 mb-2 tracking-widest flex items-center gap-1.5"><i class="fa-solid fa-lightbulb"></i> Analogy</p>
                        <p class="text-sm text-slate-700 leading-relaxed font-medium">${t.analogy}</p>
                    </div>
                </div>
            `}).join('');
        }

        // PDF Download Script
        function downloadNotes() {
            const element = document.getElementById('tab-summary');
            
            // Add temporary styling specifically for PDF export
            const originalGridClass = document.getElementById('topics-grid').className;
            document.getElementById('topics-grid').className = "flex flex-col gap-4"; // Tighter gap for PDF

            const opt = {
                margin:       [0.5, 0.5, 0.5, 0.5],
                filename:     'Kaparsh_Notes.pdf',
                image:        { type: 'jpeg', quality: 0.98 },
                html2canvas:  { scale: 2, useCORS: true, windowWidth: 1024 }, // Set window width to force desktop layout in PDF
                jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' }
            };
            
            html2pdf().set(opt).from(element).save().then(() => {
                // Restore original styling after download
                document.getElementById('topics-grid').className = originalGridClass;
            });
        }

        async function generateSchedule() {
            const examDate = document.getElementById('exam-date').value;
            const studyHours = document.getElementById('study-hours').value;
            if (!examDate) return alert("Please set an exam date.");

            const loadedTopics = AppState.topics.filter(t => t.loaded);
            toggleLoader(true, 'Structuring timeline...');

            try {
                const response = await fetch('/api/schedule', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topics: loadedTopics, exam_date: examDate, study_hours: parseFloat(studyHours) })
                });
                
                const rawText = await response.text();
                if (!response.ok) throw new Error(JSON.parse(rawText).detail || "Server Error");
                
                AppState.schedule = JSON.parse(rawText).schedule;
                renderSchedule();
                
                document.getElementById('schedule-setup').classList.add('hidden');
                document.getElementById('schedule-result').classList.remove('hidden');
                toggleLoader(false);
            } catch (err) { showError(err.message); }
        }

        function renderSchedule() {
            const container = document.getElementById('schedule-result');
            container.innerHTML = AppState.schedule.map((day) => `
                <div class="flex gap-5 p-6 bg-white rounded-2xl border border-slate-200 shadow-sm hover:border-brand-300 transition-colors">
                    <div class="flex-shrink-0 text-center pr-6 border-r border-slate-100 flex flex-col justify-center">
                        <span class="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Day ${day.day}</span>
                        <span class="text-xl font-bold text-slate-900">${day.date.split('-').slice(1).join('/')}</span>
                    </div>
                    <div class="flex-1 py-1 pl-2">
                        <div class="flex justify-between items-start mb-3">
                            <h4 class="font-bold text-slate-900 text-base">${day.focus_area}</h4>
                            <span class="text-xs font-bold text-brand-600 bg-brand-50 px-2 py-1 rounded">${day.hours_allocated} hrs</span>
                        </div>
                        <div class="flex flex-wrap gap-2 mb-4">
                            ${day.topics_to_study.map(t => `<span class="text-xs px-2.5 py-1 bg-white border border-slate-200 text-slate-700 font-semibold rounded-md shadow-sm">${t}</span>`).join('')}
                        </div>
                        <p class="text-sm text-slate-600 leading-relaxed font-medium"><i class="fa-solid fa-arrow-trend-up text-brand-400 mr-1.5"></i> ${day.actionable_advice}</p>
                    </div>
                </div>
            `).join('');
        }

        async function generateQuiz() {
            toggleLoader(true, 'Generating questions...');
            const loadedTopics = AppState.topics.filter(t => t.loaded);

            try {
                const response = await fetch('/api/quiz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topics: loadedTopics })
                });
                
                const rawText = await response.text();
                if (!response.ok) throw new Error(JSON.parse(rawText).detail || "Server Error");
                
                AppState.quiz = JSON.parse(rawText).quiz;
                renderQuiz();
                
                document.getElementById('quiz-setup').classList.add('hidden');
                document.getElementById('quiz-result').classList.remove('hidden');
                toggleLoader(false);
            } catch (err) { showError(err.message); }
        }

        function renderQuiz() {
            const container = document.getElementById('quiz-questions-container');
            document.getElementById('quiz-score-area').innerHTML = ''; 

            container.innerHTML = AppState.quiz.map((q, index) => `
                <div id="qcard-${index}" class="p-8 bg-white rounded-2xl border border-slate-200 shadow-sm relative">
                    <div class="absolute -left-3 top-8 w-7 h-7 bg-brand-600 text-white rounded-full flex items-center justify-center font-bold text-xs shadow-sm border-2 border-white">${index + 1}</div>
                    <h4 class="font-bold text-lg mb-5 text-slate-900 leading-relaxed pl-2">${q.question}</h4>
                    <div class="space-y-3">
                        ${q.options.map((opt, oIndex) => `
                            <label class="flex items-start gap-4 cursor-pointer p-4 rounded-xl border border-slate-200 hover:border-brand-400 hover:bg-brand-50 transition-all">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}" class="mt-1 w-4 h-4 text-brand-600 focus:ring-brand-500 border-slate-300">
                                <span class="text-sm text-slate-700 font-medium">${opt}</span>
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
                    resultDiv.innerHTML = `<div class="text-sm text-amber-600 font-bold py-3 px-4 bg-amber-50 rounded-xl mt-4 border border-amber-200"><i class="fa-solid fa-triangle-exclamation mr-2"></i> Please select an answer.</div>`;
                    return;
                }
                
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="p-5 bg-emerald-50 rounded-xl border border-emerald-200 mt-5">
                            <p class="text-emerald-700 font-bold text-sm mb-2 flex items-center gap-2"><i class="fa-solid fa-circle-check"></i> Correct</p>
                            <p class="text-sm text-emerald-800 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-emerald-400', 'ring-4', 'ring-emerald-50');
                    card.classList.remove('border-slate-200');
                } else {
                    resultDiv.innerHTML = `
                        <div class="p-5 bg-rose-50 rounded-xl border border-rose-200 mt-5">
                            <p class="text-rose-700 font-bold text-sm mb-2 flex items-center gap-2"><i class="fa-solid fa-circle-xmark"></i> Incorrect</p>
                            <p class="text-sm mb-4 text-slate-800 font-medium">Correct Answer: <span class="font-bold text-slate-900 bg-white px-2 py-1 rounded border border-slate-200 ml-1">${q.correct_answer}</span></p>
                            <div class="h-px w-full bg-rose-200 mb-3"></div>
                            <p class="text-[10px] uppercase font-bold text-rose-500 mb-1.5 tracking-wider">Analysis</p>
                            <p class="text-sm text-rose-800 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-rose-400', 'ring-4', 'ring-rose-50');
                    card.classList.remove('border-slate-200');
                }
            });
            
            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="mt-8 p-8 bg-slate-900 rounded-2xl text-center text-white shadow-xl">
                        <p class="text-xs font-bold uppercase tracking-widest text-brand-400 mb-2">Final Score</p>
                        <h3 class="text-5xl font-extrabold text-white mb-3">${score} <span class="text-slate-400 text-3xl">/ ${AppState.quiz.length}</span></h3>
                        <p class="text-slate-400 text-sm font-medium">Assessment Complete.</p>
                    </div>
                `;
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