# PDF Semantic Cross-Referencer

This Streamlit app compares two PDF documents by:
- Extracting and grouping text into semantic chunks.
- Tagging each chunk using Google's Gemini API.
- Normalizing tags to ensure consistent categorization.
- Detecting semantic conflicts between matching chunks across documents.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

1. **Get a Gemini API key** from [Google MakerSuite](https://makersuite.google.com/app/apikey).

2. **Set the API key** either as:
   - Environment variable:
     ```bash
     export GEMINI_API_KEY="your_api_key_here"   # macOS/Linux
     set GEMINI_API_KEY=your_api_key_here        # Windows
     ```
   - Or in `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

## Running the App

```bash
streamlit run your_file_name.py
```

Replace `your_file_name.py` with the name of the Python file containing the app.

## Usage

1. Upload two PDF files in the app interface.
2. Wait for text extraction, tagging, and conflict detection to complete.
3. View detected semantic conflicts, unrelated matches, and all tagged chunks.
4. Download the full JSON conflict report.
