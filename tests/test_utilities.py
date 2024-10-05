import pytest
from reportlab.pdfgen import canvas
import pandas as pd
from app import (preprocess_text, allowed_file, get_file_mime_type,
                 extract_text_from_pdf, extract_text_from_docx, extract_text_from_rtf,
                 calculate_keyword_score, calculate_similarity_scores, process_job_dataframe)

# Setup: A helper function to create mock files for testing
@pytest.fixture
def sample_pdf_file(tmp_path):
    sample_pdf = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(sample_pdf))
    c.drawString(100, 750, "Valid PDF Content 123")
    c.save()
    return str(sample_pdf)


@pytest.fixture
def sample_docx_file(tmp_path):
    sample_docx = tmp_path / "sample.docx"
    sample_docx.write_text("Sample DOCX Content")
    return str(sample_docx)

# --- Test File Utilities ---
def test_allowed_file():
    assert allowed_file("resume.pdf") is True
    assert allowed_file("resume.docx") is True
    assert allowed_file("resume.txt") is True
    assert allowed_file("resume.jpg") is False  # Invalid file type

def test_get_file_mime_type(sample_pdf_file):
    mime_type = get_file_mime_type(sample_pdf_file)
    assert mime_type == "application/pdf"

# --- Test Text Extraction ---
def test_extract_text_from_pdf(sample_pdf_file):
    text = extract_text_from_pdf(sample_pdf_file)
    print(text)
    assert "Valid PDF Content" in text # Ensure text is extracted

# Note: DOCX, RTF extraction functions can be similarly tested

# --- Test Text Preprocessing ---
def test_preprocess_text():
    raw_text = "Hello World!! This is a test. 123"
    processed_text = preprocess_text(raw_text)
    assert processed_text == "hello world this is a test 123"

# --- Test Job DataFrame Processing ---
def test_process_job_dataframe():
    raw_data = {
        'title': ['Developer', 'Engineer', None],
        'description': ['Coding', 'Building', None],
        'date_posted': ['2023-01-01', None, '2022-12-31'],
        'company': ['ABC Corp', None, 'XYZ Corp']
    }
    df = pd.DataFrame(raw_data)
    processed_df = process_job_dataframe(df)
    assert processed_df['title'][2] == ''  # Missing title is replaced with ''
    assert processed_df['description'][2] == ''  # Missing description
    assert processed_df['date_posted'][1] != None  # Date is filled with today's date
    assert processed_df['company'][1] == ''  # Missing company

# --- Test Similarity Calculation ---
def test_calculate_keyword_score():
    job_text = "Looking for a candidate with Python skills."
    keywords = ["Python", "Machine Learning"]
    score = calculate_keyword_score(job_text, keywords)
    assert score == 0.5  # One keyword matched

def test_calculate_similarity_scores():
    cv_text = "Python developer with experience in AI."
    job_texts = ["Looking for Python developer", "AI engineer needed"]
    keywords = ["Python", "AI"]

    tfidf_scores, sbert_scores, keyword_scores = calculate_similarity_scores(cv_text, job_texts, keywords)
    assert len(tfidf_scores) == 2  # Two job descriptions
    assert len(sbert_scores) == 2
    assert len(keyword_scores) == 2
    assert all(0 <= score <= 1 for score in tfidf_scores)  # Normalized scores
    assert all(0 <= score <= 1 for score in sbert_scores)  # Normalized scores
    assert all(0 <= score <= 1 for score in keyword_scores)  # Normalized scores
