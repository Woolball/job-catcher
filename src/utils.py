import os
import regex as re
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
from langid.langid import LanguageIdentifier, model
import stopwordsiso
import logging

# Initialize logger
logger = logging.getLogger(__name__)
lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
delimiters = r'[,;.]+'

def preprocess_text(text):
    """Efficient text preprocessing: convert to lowercase, replace non-alphanumeric chars, reduce spaces."""
    if not text:
        return ''
    # Use Unicode properties to keep letters (any language) and numbers, replacing other chars
    return re.sub(r'[^\p{L}\p{N}]+', ' ', text.lower()).strip()

def detect_language(text):
    if not text:
        return None
    lang, confidence = lang_identifier.classify(text[:500])
    return lang if confidence >= 0.3 else None

stopwords_cache = {}

def remove_stopwords(text_list, lang_code, fallback_lang='en'):
    """Remove stopwords from a list of texts based on detected language."""
    if lang_code not in stopwords_cache:
        if stopwordsiso.has_lang(lang_code):
            stopwords_cache[lang_code] = set(stopwordsiso.stopwords(lang_code))
        elif stopwordsiso.has_lang(fallback_lang):
            logger.warning(f"Stopwords for '{lang_code}' not found. Using fallback: {fallback_lang}.")
            stopwords_cache[lang_code] = set(stopwordsiso.stopwords(fallback_lang))
        else:
            logger.warning(f"Neither '{lang_code}' nor fallback '{fallback_lang}' have stopword lists.")
            return text_list

    stopwords = stopwords_cache[lang_code]

    return [
        ' '.join([word for word in text.split() if word not in stopwords])
        for text in text_list
    ]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_file_mime_type(file_path):
    return magic.from_file(file_path, mime=True)


def extract_text_from_file(file_path, mime_type):
    """Extracts text from different file formats based on MIME type."""
    try:
        extractors = {
            'application/pdf': extract_text_from_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': extract_text_from_docx,
            'application/msword': extract_text_from_docx,
            'application/rtf': extract_text_from_rtf,
            'text/plain': extract_text_from_txt
        }
        if mime_type not in extractors:
            raise ValueError(f"Unsupported file type: {mime_type}")
        return extractors[mime_type](file_path)
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {str(e)}")
        raise ValueError(f"Failed to process file: {file_path}. Error: {str(e)}")


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
    location = f"{region}, {country}" if region else country

    preferred_keywords_pure = [preprocess_text(keyword.strip()) for keyword in re.split(delimiters, form.get('preferred_keywords', ''))]
    preferred_keywords = preferred_keywords_pure + search_terms #append to the list of keywords all the search terms
    preferred_keywords += [word for phrase in preferred_keywords for word in phrase.split()] #append to the list of keywords all individual words
    preferred_keywords = list(set(preferred_keywords)) #remove duplicates
    preferred_keywords += preferred_keywords_pure #duplicate original keywords as they deserve double the weight
    preferred_keywords = [s for s in preferred_keywords if s] #remove empty strings
    required_keywords = [preprocess_text(keyword.strip()) for keyword in
                     re.split(delimiters, form.get('required_keywords', ''))]
    if required_keywords == ['']:
        required_keywords = []
    exclude_keywords = [preprocess_text(keyword.strip()) for keyword in
                     re.split(delimiters, form.get('exclude_keywords', ''))]
    if exclude_keywords == ['']:
        exclude_keywords = []
    interval = form.get('posted_since', 'month')

    # Validate that the essential text fields are not empty after preprocessing
    if not search_terms or all(not term for term in search_terms):
        return jsonify({'error': 'Job titles are required and cannot be empty or gibberish'}), 400
    if not country:
        return jsonify({'error': 'Location is required and cannot be empty or gibberish'}), 400

    # Validate and clean the uploaded file
    file = files.get('cv_file', None)
    if file:
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported CV file extension'}), 400
        else:
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
                os.remove(file_path)
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
        cv_text = ' '.join(preferred_keywords)

    # Return cleaned data
    return {
        'search_terms': search_terms,
        'location': location,
        'interval': interval,
        'cv_text': cv_text,
        'preferred_keywords': preferred_keywords,
        'required_keywords': required_keywords,
        'exclude_keywords': exclude_keywords
    }


def dump_ranked_jobs(ranked_jobs_df, file_path):
    ranked_jobs_df.to_csv(file_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)


def process_job_dataframe(jobs_df):
    if not jobs_df.empty:
        jobs_df['date_posted'] = pd.to_datetime(jobs_df['date_posted'], errors='coerce').fillna(datetime.today()).dt.strftime('%b %d')
        jobs_df['display_title'] = jobs_df['title'].fillna('').replace('', 'Unknown').str.strip()
        jobs_df['display_company'] = jobs_df['company'].fillna('').replace('', 'Unknown').str.strip().str.title()
        jobs_df['title'] = jobs_df['display_title'].apply(preprocess_text)
        jobs_df['description'] = jobs_df['description'].fillna('').apply(preprocess_text)
        jobs_df['description'] = (jobs_df['title'] + ' ' + jobs_df['description']).str.strip()
        jobs_df['company'] = jobs_df['display_company'].apply(preprocess_text)
        # Drop duplicates based on display titles and 1st word of display company
        jobs_df['first_word_company'] = jobs_df['display_company'].str.split().str[0].apply(preprocess_text)
        jobs_df = jobs_df.drop_duplicates(subset=['title', 'first_word_company'])
        jobs_df = jobs_df.drop(columns=['first_word_company'])
    return jobs_df





