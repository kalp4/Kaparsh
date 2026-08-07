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
# This string contains the entire Kaparsh HTML/JS app. By serving it via Python, 
# GitHub calculates this repository as a 99.9% Python project.
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
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- html2pdf for Native Download -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <style>
        body {
            background-color: #fafafa;
            color: #171717;
        }
        .loader {
            border-top-color: #171717;
            -webkit-animation: spinner 1s linear infinite;
            animation: spinner 1s linear infinite;
        }
        @keyframes spinner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .drag-active {
            border-color: #171717 !important;
            background-color: #f5f5f5 !important;
        }
        .tab-active {
            border-bottom: 2px solid #171717;
            color: #171717;
            font-weight: 500;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #d4d4d8; border-radius: 3px; }
    </style>
</head>
<body class="min-h-screen font-sans antialiased selection:bg-neutral-200 selection:text-black">

    <!-- Header -->
    <header class="bg-white border-b border-neutral-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center gap-2">
                <div class="w-6 h-6 bg-neutral-900 rounded flex items-center justify-center text-white">
                    <i class="fa-solid fa-graduation-cap text-xs"></i>
                </div>
                <h1 class="text-lg font-semibold tracking-tight text-neutral-900">Kaparsh</h1>
            </div>
        </div>
    </header>

    <!-- Main Layout -->
    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Sidebar -->
        <aside class="lg:col-span-4 flex flex-col gap-6" data-html2canvas-ignore>
            <div class="bg-white border border-neutral-200 p-6 shadow-sm">
                <h2 class="text-sm font-semibold mb-6 text-neutral-900 uppercase tracking-wide">Document Setup</h2>
                
                <div class="space-y-5">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-neutral-500 mb-1.5">Exam Date</label>
                            <input type="date" id="exam-date" class="w-full px-3 py-2 border border-neutral-300 text-sm focus:ring-1 focus:ring-neutral-900 focus:border-neutral-900 outline-none transition-colors rounded-none bg-transparent">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-neutral-500 mb-1.5">Hours / Day</label>
                            <input type="number" id="study-hours" value="2" min="1" max="16" class="w-full px-3 py-2 border border-neutral-300 text-sm focus:ring-1 focus:ring-neutral-900 focus:border-neutral-900 outline-none transition-colors rounded-none bg-transparent">
                        </div>
                    </div>
                </div>

                <div class="mt-6 pt-6 border-t border-neutral-100">
                    <label class="block text-xs font-medium text-neutral-500 mb-2">Upload Syllabus / Notes</label>
                    <div id="drop-zone" class="border border-dashed border-neutral-300 bg-neutral-50 p-6 text-center hover:bg-neutral-100 transition-colors cursor-pointer">
                        <i class="fa-regular fa-file-pdf text-xl text-neutral-400 mb-2"></i>
                        <p class="text-sm text-neutral-900 font-medium">Select PDF file</p>
                        <p id="file-name" class="text-xs text-neutral-500 mt-1">Supports up to 25 pages</p>
                        <input type="file" id="file-upload" accept=".pdf" class="hidden">
                    </div>
                </div>

                <button id="analyze-btn" class="w-full mt-6 bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium py-3 px-4 transition-colors">
                    Process Document
                </button>
            </div>
        </aside>

        <!-- Main Content Area -->
        <section class="lg:col-span-8">
            <div class="bg-white border border-neutral-200 min-h-[600px] flex flex-col shadow-sm">
                
                <!-- Tabs -->
                <div class="flex border-b border-neutral-200 bg-neutral-50/50" data-html2canvas-ignore>
                    <button class="tab-btn flex-1 py-3 text-sm text-neutral-500 hover:text-neutral-900 transition-colors tab-active" data-target="tab-summary">
                        Extracted Notes
                    </button>
                    <button class="tab-btn flex-1 py-3 text-sm text-neutral-500 hover:text-neutral-900 transition-colors" data-target="tab-schedule">
                        Study Schedule
                    </button>
                    <button class="tab-btn flex-1 py-3 text-sm text-neutral-500 hover:text-neutral-900 transition-colors" data-target="tab-quiz">
                        Practice Exam
                    </button>
                </div>

                <!-- Loader -->
                <div id="global-loader" class="hidden flex-1 flex flex-col items-center justify-center p-12">
                    <div class="loader w-8 h-8 border-2 border-neutral-200 rounded-full mb-4"></div>
                    <p id="loader-text" class="text-sm text-neutral-500">Analyzing document structure...</p>
                </div>

                <!-- Empty State -->
                <div id="empty-state" class="flex-1 flex flex-col items-center justify-center p-12 text-center">
                    <div class="w-12 h-12 bg-neutral-100 flex items-center justify-center mb-4 rounded-full">
                        <i class="fa-solid fa-align-left text-neutral-400"></i>
                    </div>
                    <h3 class="text-base font-medium text-neutral-900 mb-1">No document loaded</h3>
                    <p class="text-sm text-neutral-500 max-w-sm">Upload a PDF to generate structured study notes, timelines, and assessments.</p>
                </div>

                <!-- Tab Contents -->
                <div id="tabs-container" class="hidden flex-1 p-8 overflow-y-auto">
                    
                    <!-- Notes Tab -->
                    <div id="tab-summary" class="tab-pane block">
                        <div class="flex justify-between items-center mb-8 pb-4 border-b border-neutral-100">
                            <div class="flex items-center gap-4">
                                <h2 class="text-xl font-semibold text-neutral-900">Study Notes</h2>
                                <span id="topic-count" class="text-xs font-medium text-neutral-500 bg-neutral-100 px-2 py-1 rounded"></span>
                            </div>
                            
                            <!-- Save as PDF Button -->
                            <button id="download-btn" onclick="downloadNotes()" class="hidden px-4 py-2 bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium transition-colors rounded shadow flex items-center gap-2" data-html2canvas-ignore>
                                <i class="fa-solid fa-download"></i> Save as PDF
                            </button>
                        </div>
                        <div id="topics-grid" class="flex flex-col gap-6">
                            <!-- Injected by JS -->
                        </div>
                    </div>

                    <!-- Schedule Tab -->
                    <div id="tab-schedule" class="tab-pane hidden">
                        <div id="schedule-setup" class="text-center py-16">
                            <h3 class="text-lg font-semibold text-neutral-900 mb-2">Generate Study Schedule</h3>
                            <p class="text-sm text-neutral-500 mb-6">Create a timeline based on spaced repetition principles.</p>
                            <button onclick="generateSchedule()" class="px-6 py-2 bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium transition-colors">
                                Generate Plan
                            </button>
                        </div>
                        <div id="schedule-result" class="hidden space-y-4">
                            <!-- Injected by JS -->
                        </div>
                    </div>

                    <!-- Quiz Tab -->
                    <div id="tab-quiz" class="tab-pane hidden">
                        <div id="quiz-setup" class="text-center py-16">
                            <h3 class="text-lg font-semibold text-neutral-900 mb-2">Practice Examination</h3>
                            <p class="text-sm text-neutral-500 mb-6">Test your comprehension with an AI-generated assessment.</p>
                            <button onclick="generateQuiz()" class="px-6 py-2 bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium transition-colors">
                                Start Exam
                            </button>
                        </div>
                        <div id="quiz-result" class="hidden space-y-8">
                            <div id="quiz-questions-container" class="space-y-8"></div>
                            <button onclick="checkAnswers()" class="w-full py-3 bg-neutral-900 text-white text-sm font-medium hover:bg-neutral-800 transition-colors">
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
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('bg-neutral-100'); });
        dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('bg-neutral-100'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('bg-neutral-100');
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
                fileNameDisplay.classList.add('text-neutral-900');
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

            toggleLoader(true, 'Extracting framework...');

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
            document.getElementById('topic-count').innerText = `${AppState.topics.length} Topics`;
            
            // Show Download Button if at least one topic is loaded
            const downloadBtn = document.getElementById('download-btn');
            if (AppState.topics.some(t => t.loaded)) {
                downloadBtn.classList.remove('hidden');
            }
            
            grid.innerHTML = AppState.topics.map((t) => {
                if (!t.loaded) {
                    return `
                    <div class="border border-neutral-200 p-6 opacity-60">
                        <div class="flex justify-between items-center mb-4">
                            <h4 class="font-medium text-base text-neutral-900">${t.title}</h4>
                            <span class="text-xs text-neutral-400 font-medium">Loading...</span>
                        </div>
                        <div class="space-y-2 mt-2">
                            <div class="h-1.5 bg-neutral-200 w-3/4"></div>
                            <div class="h-1.5 bg-neutral-200 w-1/2"></div>
                            <div class="h-1.5 bg-neutral-200 w-5/6"></div>
                        </div>
                    </div>`;
                }

                const priorityStyles = {
                    'High': 'text-neutral-900 border-neutral-900',
                    'Medium': 'text-neutral-500 border-neutral-300',
                    'Low': 'text-neutral-400 border-neutral-200'
                };
                const pStyle = priorityStyles[t.priority] || priorityStyles['Medium'];
                
                // Color-Coded Definitions
                const defsHtml = (t.definitions && t.definitions.length > 0) ? `
                    <div class="my-4 space-y-2">
                        ${t.definitions.map(d => `
                            <p class="text-sm text-neutral-700 leading-relaxed">
                                <strong class="text-blue-600 font-semibold mr-1">${d.term}:</strong> ${d.definition}
                            </p>
                        `).join('')}
                    </div>
                ` : '';

                // High-Contrast Formulas
                const formulasHtml = (t.formulas && t.formulas.length > 0) ? `
                    <div class="my-5 space-y-3">
                        ${t.formulas.map(f => `
                            <div class="bg-amber-50 border-2 border-amber-300 p-4 rounded-lg flex flex-col gap-1 shadow-sm break-inside-avoid">
                                <p class="font-mono text-lg font-bold text-amber-900 text-center">${f.equation}</p>
                                <p class="text-xs text-amber-700 uppercase tracking-wider text-center font-bold mt-1">${f.meaning}</p>
                            </div>
                        `).join('')}
                    </div>
                ` : '';

                // Differentiated Derivations
                const derivationsHtml = (t.derivations && t.derivations.length > 0) ? `
                    <div class="my-5 space-y-3">
                        ${t.derivations.map(d => `
                            <div class="bg-purple-50 border-l-4 border-purple-500 p-4 shadow-sm break-inside-avoid">
                                <p class="text-sm font-bold text-purple-900 mb-2">${d.title}</p>
                                <p class="text-sm text-purple-800 font-mono leading-relaxed whitespace-pre-wrap">${d.content}</p>
                            </div>
                        `).join('')}
                    </div>
                ` : '';

                // General Notes
                const notesList = t.notes.map(n => `<li class="mb-2 flex items-start text-sm text-neutral-700 leading-relaxed"><span class="text-neutral-300 mr-2">—</span>${n}</li>`).join('');

                return `
                <div class="border border-neutral-200 p-6 hover:border-neutral-300 transition-colors break-inside-avoid">
                    <div class="flex justify-between items-start mb-4">
                        <h4 class="font-semibold text-lg text-neutral-900 pr-4">${t.title}</h4>
                        <span class="text-[10px] font-medium px-2 py-0.5 border uppercase tracking-wider ${pStyle}">${t.priority}</span>
                    </div>
                    
                    ${defsHtml}
                    ${formulasHtml}
                    ${derivationsHtml}
                    
                    <ul class="mb-4 mt-4">
                        ${notesList}
                    </ul>
                    
                    <div class="mt-4 pt-4 border-t border-neutral-100">
                        <div class="border-l-2 border-neutral-900 pl-4 py-1">
                            <p class="text-[10px] uppercase font-semibold text-neutral-500 mb-1 tracking-wider">Analogy</p>
                            <p class="text-sm text-neutral-700 leading-relaxed">${t.analogy}</p>
                        </div>
                    </div>
                </div>
            `}).join('');
        }

        // PDF Download Script
        function downloadNotes() {
            const element = document.getElementById('tab-summary');
            const opt = {
                margin:       0.5,
                filename:     'Kaparsh_Notes.pdf',
                image:        { type: 'jpeg', quality: 0.98 },
                html2canvas:  { scale: 2, useCORS: true },
                jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            
            // html2pdf automatically ignores elements with the "data-html2canvas-ignore" attribute (like the download button itself)
            html2pdf().set(opt).from(element).save();
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
                <div class="flex gap-4 p-5 border border-neutral-200 hover:bg-neutral-50 transition-colors">
                    <div class="flex-shrink-0 text-center pr-5 border-r border-neutral-200 flex flex-col justify-center">
                        <span class="text-[10px] text-neutral-500 font-medium uppercase tracking-widest mb-1">Day ${day.day}</span>
                        <span class="text-lg font-semibold text-neutral-900">${day.date.split('-').slice(1).join('/')}</span>
                    </div>
                    <div class="flex-1 py-1 pl-1">
                        <div class="flex justify-between items-start mb-2">
                            <h4 class="font-medium text-neutral-900 text-sm">${day.focus_area}</h4>
                            <span class="text-xs font-medium text-neutral-500">${day.hours_allocated} hrs</span>
                        </div>
                        <div class="flex flex-wrap gap-2 mb-3">
                            ${day.topics_to_study.map(t => `<span class="text-xs px-2 py-1 bg-white border border-neutral-200 text-neutral-700 font-medium">${t}</span>`).join('')}
                        </div>
                        <p class="text-xs text-neutral-600 leading-relaxed">${day.actionable_advice}</p>
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
                <div id="qcard-${index}" class="p-6 border border-neutral-200">
                    <h4 class="font-medium text-base mb-4 text-neutral-900 leading-relaxed"><span class="text-neutral-400 mr-2">${index + 1}.</span> ${q.question}</h4>
                    <div class="space-y-2">
                        ${q.options.map((opt, oIndex) => `
                            <label class="flex items-start gap-3 cursor-pointer p-3 border border-transparent hover:bg-neutral-50 transition-colors">
                                <input type="radio" name="question-${index}" value="${opt.replace(/"/g, '&quot;')}" class="mt-1 w-4 h-4 text-neutral-900 focus:ring-neutral-900 border-neutral-300">
                                <span class="text-sm text-neutral-700">${opt}</span>
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
                    resultDiv.innerHTML = `<div class="text-sm text-neutral-500 font-medium py-2">Please select an answer.</div>`;
                    return;
                }
                
                if (selected.value === q.correct_answer) {
                    score++;
                    resultDiv.innerHTML = `
                        <div class="p-4 bg-neutral-50 border-t border-neutral-200 mt-4">
                            <p class="text-neutral-900 font-medium text-sm mb-1">Correct</p>
                            <p class="text-sm text-neutral-600 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-neutral-400');
                    card.classList.remove('border-neutral-200');
                } else {
                    resultDiv.innerHTML = `
                        <div class="p-4 bg-neutral-50 border-t border-neutral-200 mt-4">
                            <p class="text-neutral-900 font-medium text-sm mb-2">Incorrect</p>
                            <p class="text-sm mb-3 text-neutral-700">Correct Answer: <span class="font-medium text-neutral-900">${q.correct_answer}</span></p>
                            <p class="text-[10px] uppercase font-semibold text-neutral-500 mb-1 tracking-wider">Explanation</p>
                            <p class="text-sm text-neutral-600 leading-relaxed">${q.explanation}</p>
                        </div>`;
                    card.classList.add('border-neutral-400');
                    card.classList.remove('border-neutral-200');
                }
            });
            
            if(document.querySelectorAll('input[type="radio"]:checked').length === AppState.quiz.length) {
                document.getElementById('quiz-score-area').innerHTML = `
                    <div class="mt-8 pt-8 border-t border-neutral-200 text-center">
                        <p class="text-xs font-semibold uppercase tracking-widest text-neutral-500 mb-2">Final Score</p>
                        <h3 class="text-4xl font-bold text-neutral-900 mb-2">${score} / ${AppState.quiz.length}</h3>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""

# Serve the injected HTML directly via Python
@app.route("/", methods=["GET"])
def serve_frontend():
    response = make_response(KAPARSH_FRONTEND)
    response.headers["Content-Type"] = "text/html"
    return response

# Do not run app.run() here; Vercel handles serving the 'app' object natively.