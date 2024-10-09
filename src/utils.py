import os
import re
import magic
from flask import jsonify
from config import Config
import pandas as pd
import csv
from datetime import datetime
from werkzeug.utils import secure_filename
import pdfplumber
import docx
import pypandoc
import logging

# Initialize logger
logger = logging.getLogger(__name__)


delimiters = r'[^a-zA-Z\s]+'


def preprocess_text(text):
    if not text:
        return ''
    return ''.join(c for c in text.lower().replace('\n', ' ') if c.isalnum() or c.isspace())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_file_mime_type(file_path):
    return magic.from_file(file_path, mime=True)


def extract_text_from_file(file_path, mime_type):
    """Extracts text from different file formats based on MIME type and limits it to 2000 words."""
    if mime_type == 'application/pdf':
        text = extract_text_from_pdf(file_path)
    elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
        text = extract_text_from_docx(file_path)
    elif mime_type == 'application/rtf':
        text = extract_text_from_rtf(file_path)
    elif mime_type == 'text/plain':
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

    return text


def extract_text_from_pdf(file_path):
    """Extract text from a PDF using pdfplumber."""
    with pdfplumber.open(file_path) as pdf:
        return ' '.join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_rtf(file_path):
    """Extract text from an RTF file using pypandoc."""
    return pypandoc.convert_file(file_path, 'plain')

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_and_clean_input(form, files):
    """Validates and cleans form inputs and file uploads."""

    # Extract and preprocess text fields
    search_terms = [preprocess_text(term.strip()) for term in re.split(delimiters, form.get('search_terms', ''))]
    search_terms = [s for s in search_terms if s]
    country = form.get('country', '').strip().split(',')[0]
    region = form.get('region', '').strip().split(',')[0]
    if region:
        location = region + ', ' + country
    else:
        location = country
    keywords_pure = [preprocess_text(keyword.strip()) for keyword in re.split(delimiters, form.get('keywords', ''))]
    keywords = keywords_pure + search_terms #append to the list of keywords all the search terms
    keywords += [word for phrase in keywords for word in phrase.split()] #append to the list of keywords all individual words
    keywords = list(set(keywords)) #remove duplicates
    keywords += keywords_pure #duplicate original keywords as they deserve double the weight
    keywords = [s for s in keywords if s] #remove empty strings
    interval = form.get('posted_since', 'month')

    # Validate that the essential text fields are not empty after preprocessing
    if not search_terms or all(not term for term in search_terms):
        return jsonify({'error': 'Job titles are required and cannot be empty or gibberish'}), 400
    if not country:
        return jsonify({'error': 'Location is required and cannot be empty or gibberish'}), 400

    # Validate and clean the uploaded file
    file = files.get('cv_file', None)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Detect MIME type
        mime_type = get_file_mime_type(file_path)
        if mime_type not in Config.ALLOWED_MIME_TYPES:
            os.remove(file_path)
            return jsonify({'error': 'Unsupported CV file type'}), 400

        # Extract and preprocess text from file
        try:
            raw_cv_text = extract_text_from_file(file_path, mime_type)
            os.remove(file_path)  # Clean up the file after extracting the text
        except Exception as e:
            os.remove(file_path)
            logger.error(f"Error processing CV file: {filename}. Exception: {str(e)}")
            return jsonify({'error': f'Error processing CV file: {filename}'}), 500

        cv_text = preprocess_text(raw_cv_text)
        cv_text = ' '.join(cv_text.split()[:Config.CV_TEXT_LIMIT]) # limit cv text

        # Check if CV text is empty after preprocessing
        if not cv_text:
            return jsonify({'error': 'CV file is empty or contains invalid content'}), 400
    else:
        cv_text = ' '.join(keywords)

    # Return cleaned data
    return {
        'search_terms': search_terms,
        'location': location,
        'interval': interval,
        'keywords': keywords,
        'cv_text': cv_text
    }


def dump_ranked_jobs(ranked_jobs_df, file_path):
    ranked_jobs_df.to_csv(file_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)


def process_job_dataframe(jobs_df):
    if not jobs_df.empty:
        jobs_df['date_posted'] = jobs_df['date_posted'].fillna(datetime.today().strftime('%Y-%m-%d'))
        jobs_df['date_posted'] = jobs_df['date_posted'].apply(
            lambda date_value: pd.to_datetime(date_value, errors='coerce').strftime("%b %d"))
        jobs_df['display_title'] = jobs_df['title'].fillna('').str.strip()
        jobs_df['display_company'] = jobs_df['company'].fillna('').str.strip().str.title()
        jobs_df['title'] = jobs_df['display_title'].apply(preprocess_text)
        jobs_df['description'] = jobs_df['description'].fillna('').apply(preprocess_text)
        jobs_df['description'] = jobs_df['title'] + ' ' + jobs_df['description']
        jobs_df['company'] = jobs_df['display_company'].apply(preprocess_text)
    return jobs_df





