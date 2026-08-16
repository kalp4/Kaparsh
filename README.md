# 📚 Kaparsh: Your Late-Night AI Study Buddy

Let’s be real for a second: staring down a dense, 50-page textbook chapter while trying to guess what will *actually* be on the test is exhausting. You highlight a whole page, zone out, and suddenly realize you haven't retained a single word. 

We built **Kaparsh** to solve exactly that. Think of it as a smart, tireless study companion that takes your heavy, raw study materials and turns them into clean notes, beautiful formula sheets, and custom practice quizzes. 

The goal? To help you actually spend your time *learning* instead of just aggressively highlighting pages at 2 AM. 🌙

---

## ✨ What it actually does

### 1. Notes & Formulas
Upload a PDF chapter or Upload a picture of your textbook. Tell the app your current grade and subject, and let it do its thing. The AI reads through the text and breaks it down into:
*   **High-Importance Notes:** Short, punchy bullet points of the concepts that actually matter.
*   **Formula Sheets:** Math and physics equations pulled out and beautifully formatted.
*   **Step-by-Step Derivations:** Proofs are kept in their own separate section so they don't clutter your main notes.
*   **Smart Syllabus Tagging:** If the AI spots random trivia that isn't usually in your specific curriculum, it tucks it away under an `[EXTRA]` dropdown. You instantly know what to prioritize for the exam.

### 2. MCQ based Quiz to test out what you studied.
Because just reading notes isn't always enough to make things stick. Kaparsh automatically generates a multiple-choice quiz based on the exact notes it just made for you. You can take the practice test right in the app, and it gives you instant feedback—explaining exactly *why* your answer was right or wrong.

### 3. A Doubt solver
Got stuck on a tricky concept? Ask the built-in chat tutor. Instead of just spitting out generic web-search answers, this chatbot is strictly locked to your uploaded document. It answers your questions using the exact context and vocabulary from your own textbook.

### 4. It Looks Amazing
It has a fun User interface so the students would not be bored or annoyed studying. It has both light mode and dark mode so the the students can study at night without straining their eyes.

---

## 🚀 Get It Running on Your Machine

Want to try it out? You can get this running locally in just a few minutes. 

**1. Clone the repository**
```bash
git clone https://github.com/kalp4/kaparsh.git
cd kaparsh
```

**2. Install the dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your AI Brain**
Create a `.env` file in the root folder of the project and drop your Google Gemini API key in there:
```env
GEMINI_API_KEY=your_api_key_here
```

**4. Fire it up!**
```bash
flask --app main run
```

*Open up your browser, head to `http://127.0.0.1:5000`*


Incase it doesnt work, you can check out the Webapp on http://Kaparsh.vercel.app
