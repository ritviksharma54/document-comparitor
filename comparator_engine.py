import fitz  # PyMuPDF
import difflib
import re
import os
from pprint import pprint

# --- Imports needed for PDF Generation and Testing ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- V2.2 CORE COMPARISON LOGIC ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True) # Sort helps with reading order
            return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# NEW: Intelligent text chunking
def _chunk_text_intelligently(text: str) -> list[str]:
    """
    Splits text into meaningful paragraphs using a hybrid heuristic.
    A paragraph break is detected by a blank line OR a line ending with a period.
    """
    paragraphs = []
    current_paragraph = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: # Hard break for blank lines
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        current_paragraph.append(line)
        # Break if the line ends with a period and isn't part of an abbreviation (like etc.)
        if line.endswith('.') and not line.endswith(('etc.', 'i.e.', 'e.g.')):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

    # Add the last remaining paragraph
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    # If the above fails to produce many paragraphs, fall back to double-newline split
    if len(paragraphs) < 5: # Failsafe for documents with no periods or blank lines
        paragraphs = [p.replace('\n', ' ').strip() for p in text.split('\n\n') if p.strip()]

    return paragraphs

def normalize_chunk(text: str) -> str:
    """Cleans a chunk of text to make it comparable."""
    text = text.lower()
    noise_patterns = [
        r'the gazette of india', r'extraordinary', r'part ii—', r'sec. 1]',
        r'new delhi,', r'[\s\S]*?1409070/2020/mvl section', r'\(saka\)',
        r'jftlvªh lañ', r'vlk/kkj.k', r'hkkx ii', r'izkf/kdkj ls izdkf\'kr',
        r'published by authority', r'\[\d+th.*?, \d{4}\]',
        r'registered no\. dl.*'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation for matching
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_word_level_diff(text1: str, text2: str) -> tuple:
    """Compares two strings word by word."""
    words1 = re.split(r'(\s+)', text1)
    words2 = re.split(r'(\s+)', text2)
    matcher = difflib.SequenceMatcher(None, words1, words2)
    diff1, diff2 = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk1, chunk2 = "".join(words1[i1:i2]), "".join(words2[j1:j2])
        if tag == 'equal': diff1.append(('equal', chunk1)); diff2.append(('equal', chunk2))
        elif tag == 'insert': diff2.append(('insert', chunk2))
        elif tag == 'delete': diff1.append(('delete', chunk1))
        elif tag == 'replace': diff1.append(('delete', chunk1)); diff2.append(('insert', chunk2))
    return diff1, diff2

def compare_pdfs(pdf_path1: str, pdf_path2: str) -> list:
    """Compares two PDFs using the intelligent chunking and matching algorithm."""
    text1 = extract_text_from_pdf(pdf_path1)
    text2 = extract_text_from_pdf(pdf_path2)
    
    raw_paras1 = _chunk_text_intelligently(text1)
    raw_paras2 = _chunk_text_intelligently(text2)
    
    cleaned_paras1 = [normalize_chunk(p) for p in raw_paras1]
    cleaned_paras2 = [normalize_chunk(p) for p in raw_paras2]
    
    matcher = difflib.SequenceMatcher(None, cleaned_paras1, cleaned_paras2, autojunk=False)
    diff_result = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2): diff_result.append(('equal', raw_paras1[i]))
        elif tag == 'insert':
            for i in range(j1, j2): diff_result.append(('insert', raw_paras2[i]))
        elif tag == 'delete':
            for i in range(i1, i2): diff_result.append(('delete', raw_paras1[i]))
        elif tag == 'replace':
            num_paras = max(i2 - i1, j2 - j1)
            for i in range(num_paras):
                para1 = raw_paras1[i1 + i] if (i1 + i) < i2 else ""
                para2 = raw_paras2[j1 + i] if (j1 + i) < j2 else ""
                if not para1: diff_result.append(('insert', para2))
                elif not para2: diff_result.append(('delete', para1))
                else:
                    rich_diff1, rich_diff2 = get_word_level_diff(para1, para2)
                    diff_result.append(('replace', rich_diff1, rich_diff2))
    return diff_result

# --- PDF GENERATION LOGIC (UNCHANGED) ---
def create_diff_pdf(diff_result: list, output_filename: str):
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter
    margin, gutter = 0.75 * inch, 0.25 * inch
    col_width = (width - 2 * margin - gutter) / 2
    base_style = ParagraphStyle('base', fontName='Courier', fontSize=8, leading=10, alignment=TA_LEFT)
    delete_style, insert_style = "<font backColor='#fdb8c0'><strike>", "<font backColor='#a6f2b9'>"
    y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Original")
    c.drawString(margin + col_width + gutter, y, "Changed")
    y -= 0.3 * inch
    c.line(margin, y + 0.1 * inch, width - margin, y + 0.1 * inch)

    def draw_line_pair(left_text, right_text, line_type_for_bg):
        nonlocal y
        p_left = Paragraph(left_text, style=base_style)
        p_right = Paragraph(right_text, style=base_style)
        left_h, right_h = p_left.wrap(col_width, height)[1], p_right.wrap(col_width, height)[1]
        line_height = max(left_h, right_h, 10) + 4
        if y - line_height < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Original")
            c.drawString(margin + col_width + gutter, y, "Changed")
            y -= 0.3 * inch
            c.line(margin, y + 0.1 * inch, width - margin, y + 0.1 * inch)
        if line_type_for_bg == 'delete': c.setFillColor("#ffeef0"); c.rect(margin, y - line_height, col_width, line_height, stroke=0, fill=1)
        elif line_type_for_bg == 'insert': c.setFillColor("#e6ffed"); c.rect(margin + col_width + gutter, y - line_height, col_width, line_height, stroke=0, fill=1)
        p_left.drawOn(c, margin + 2, y - line_height + 2)
        p_right.drawOn(c, margin + col_width + gutter + 2, y - line_height + 2)
        y -= line_height + 2

    for line_type, *content in diff_result:
        left_line, right_line = "", ""
        def escape(text): return text.replace('&', '&').replace('<', '<').replace('>', '>')
        if line_type == 'equal': left_line = right_line = escape(content[0] or '')
        elif line_type == 'delete': left_line = f"{delete_style}{escape(content[0] or '')}</strike></font>"
        elif line_type == 'insert': right_line = f"{insert_style}{escape(content[0] or '')}</font>"
        elif line_type == 'replace':
            for tag, text in content[0]: left_line += f"{delete_style}{escape(text)}</strike></font>" if tag == 'delete' else escape(text)
            for tag, text in content[1]: right_line += f"{insert_style}{escape(text)}</font>" if tag == 'insert' else escape(text)
        draw_line_pair(left_line, right_line, line_type)
    c.save()

# --- STANDALONE TESTING BLOCK (UNCHANGED) ---
if __name__ == "__main__":
    print("--- Running Standalone Test for Comparator Engine V2.2 ---")
    original_text = "Clause 1: The work is website design.\nLine 2 is here.\n\nClause 3: The deadline is 30 days."
    changed_text = "Clause 1: The work is website design and mobile app development. Line 2 is here.\n\nClause 2: Payment will be $1200.\n\nClause 3: The deadline is 45 days from the start date."
    def create_dummy_pdf(filename, content):
        c = canvas.Canvas(filename)
        text_object = c.beginText(inch, 10 * inch)
        text_object.setFont("Helvetica", 10)
        for line in content.split('\n'): text_object.textLine(line)
        c.drawText(text_object)
        c.save()
    pdf1, pdf2, output_pdf = "test_original.pdf", "test_changed.pdf", "test_diff_output.pdf"
    create_dummy_pdf(pdf1, original_text)
    create_dummy_pdf(pdf2, changed_text)
    print("\nComparing PDFs...")
    diff_data = compare_pdfs(pdf1, pdf2)
    print("Comparison complete.")
    pprint(diff_data)
    print(f"\nGenerating comparison PDF: {output_pdf}...")
    create_diff_pdf(diff_data, output_pdf)
    print("PDF generated successfully.")
    os.remove(pdf1)
    os.remove(pdf2)
    print("\nCleanup complete. Check test_diff_output.pdf for the result.")