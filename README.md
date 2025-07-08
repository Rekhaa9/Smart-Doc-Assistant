🧠 Smart DOC Assistant

A lightweight, open-source GenAI-powered assistant that helps users interact with their PDF or TXT documents by providing summaries, Q&A, and logic-based challenge questions 

 Features

- Upload PDF or TXT documents  
- Get instant AI-generated summaries  
- Ask questions based on the document  
- Challenge yourself with logic-based questions  

 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rekhaa9/smart-doc-assistant.git
   cd smart-doc-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```
smart-doc-assistant/
├── app.py                      # Streamlit app entry point
├── requirements.txt            # Dependencies
├── utils/
│   ├── document_loader.py      # File parsing (PDF/TXT)
│   ├── summary.py              # Text summarization logic
│   ├── qa.py                   # Question answering logic
│   └── challenge.py            # Challenge question generator
```

 Technologies Used

- Streamlit
- PyPDF2
- Transformers
- Sentence-Transformers

## 👩‍💻 Author

**Rekha Kumari**

## 📃 License

MIT License


