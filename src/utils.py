import os
import regex as re
from urllib.parse import urlparse
import magic
from flask import jsonify
from config import Config
import pandas as pd
import csv
from werkzeug.utils import secure_filename
import pdfplumber
import docx
import pypandoc
from langid.langid import LanguageIdentifier, model
import stopwordsiso
import logging
from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import ratio
from google.cloud import translate_v2 as translate
from datetime import datetime, timedelta, timezone


# Initialize logger
logger = logging.getLogger(__name__)
lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
delimiters = r'[,;.]+'

# Google Translate setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_APPLICATION_CREDENTIALS
translate_client = translate.Client()
DIGIT_PLACEHODER = r'5'

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
    country_name, country_code = form.get('country', '').strip().split(',')
    region = form.get('region', '').strip().split(',')[0]
    #location = f"{region}, {country}" if region else country

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
    if not country_name:
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
        'country': {'name': country_name, 'code': country_code},
        'region': region,
        #'location': location,
        'interval': interval,
        'cv_text': cv_text,
        'preferred_keywords': preferred_keywords,
        'required_keywords': required_keywords,
        'exclude_keywords': exclude_keywords
    }


def dump_ranked_jobs(ranked_jobs_df, file_path):
    ranked_jobs_df.to_csv(file_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)


def translate_text(text, target_language='en'):
    """
    Translates text into the target language using Google Cloud Translation.
    """
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']


def strip_digits(text):
    """Remove digits from a string and replace with a placeholder."""
    return re.sub(r'\d+', DIGIT_PLACEHODER, text)


def restore_digits(template, original):
    """Restore digits from the original string into the template."""
    digits = re.findall(r'\d+', original)
    return re.sub(DIGIT_PLACEHODER, lambda _: digits.pop(0), template, count=len(digits))


def batch_translate_posted_at(strings, dest_language="en"):
    """
    Batch translate unique strings by collapsing them into categories to minimize API calls.
    """
    # Create a map of stripped strings to originals
    stripped_map = {strip_digits(s): [] for s in strings}
    for s in strings:
        stripped_map[strip_digits(s)].append(s)

    # Translate only the unique stripped strings
    stripped_strings = list(stripped_map.keys())
    translated_map = {}
    for stripped in stripped_strings:
        try:
            translated_text = translate_text(stripped, target_language=dest_language)
            translated_map[stripped] = translated_text
        except Exception as e:
            logger.error(f"Error translating {stripped}: {e}")
            translated_map[stripped] = stripped  # Fallback to the original if translation fails

    # Map back the translations to the originals with digits restored
    translations = {}
    for stripped, originals in stripped_map.items():
        for original in originals:
            translations[original] = restore_digits(translated_map[stripped], original)

    return translations


def devise_date_from_human_readable(jobs_df, human_readable_date_column_name, date_column_name):
    # Batch translation of date strings
    translations = batch_translate_posted_at(jobs_df[human_readable_date_column_name], dest_language="en")

    # Add a column for translated date strings
    jobs_df['translated_date'] = jobs_df[human_readable_date_column_name].map(translations)

    # Date calculation
    today = datetime.now(timezone.utc)

    def calculate_date(translated_date):
        number = 1
        if isinstance(translated_date, str):
            number_match = re.search(r'\d+', translated_date)
            if number_match:
                number = int(number_match.group())

            if "day" in translated_date:
                return today - timedelta(days=number)
            elif "hour" in translated_date:
                return today - timedelta(hours=number)
            elif "month" in translated_date:
                return today - timedelta(days=number * 30)
            elif "minute" in translated_date:
                return today - timedelta(minutes=number)
        return today

    # Apply the calculation to the DataFrame
    jobs_df[date_column_name] = jobs_df['translated_date'].apply(calculate_date)

    # Drop the intermediate column if not needed
    jobs_df.drop(columns=['translated_date'], inplace=True)

    return jobs_df


def filter_jobs_by_date(jobs_df, day_interval, date_column_name):
    today = datetime.now(timezone.utc)
    cutoff_date = today - timedelta(days=day_interval)
    filtered_jobs = jobs_df[pd.to_datetime(jobs_df[date_column_name], errors='coerce') >= cutoff_date]
    return filtered_jobs.reset_index(drop=True)


def extract_domain(url):
    """Extracts the domain name from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.split('.')[-2]  # Get the domain name
    return domain.capitalize()


def clean_url(url):
    """Removes query parameters and unnecessary subdomains from the URL."""
    parsed_url = urlparse(url)
    # Handle Indeed and Talent sites, preserving the first query parameter
    if 'indeed' in parsed_url.netloc or 'talent' in parsed_url.netloc:
        query_params = parsed_url.query.split('&')
        first_param = query_params[0] if query_params else ''
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        return f"{base_url}?{first_param}" if first_param else base_url
    # Check if the domain is LinkedIn and remove subdomains like 'ch.'
    if 'linkedin' in parsed_url.netloc:
        netloc_parts = parsed_url.netloc.split('.')
        if len(netloc_parts) > 2:  # Remove subdomain if it exists
            netloc = '.'.join(netloc_parts[-2:])  # Keep only the main domain and TLD
            return f"{parsed_url.scheme}://{netloc}{parsed_url.path}"
    # For other URLs, simply remove query parameters
    return url.split('?')[0]


def process_job_dataframe(jobs_df):
    if not jobs_df.empty:
        # Format date and preprocess columns
        #jobs_df['date_posted'] = pd.to_datetime(jobs_df['date_posted'], errors='coerce').fillna(pd.Timestamp(datetime.now())).dt.strftime('%b %d')
        jobs_df['date_posted'] = pd.to_datetime(jobs_df['date_posted'], errors='coerce')
        jobs_df.loc[jobs_df['date_posted'].isna(), 'date_posted'] = pd.Timestamp.today(tz='UTC')
        jobs_df['date_posted'] = pd.to_datetime(jobs_df['date_posted'], errors='coerce')
        jobs_df['date_posted'] = jobs_df['date_posted'].dt.strftime('%b %d')
        jobs_df['display_title'] = jobs_df['title'].fillna('').replace('', 'Unknown').str.strip()
        jobs_df['display_company'] = jobs_df['company'].fillna('').replace('', 'Unknown').str.strip().str.title()
        jobs_df['title'] = jobs_df['display_title'].apply(preprocess_text)
        jobs_df['description'] = jobs_df['description'].fillna('').apply(preprocess_text)
        jobs_df['description'] = (jobs_df['title'] + ' ' + jobs_df['description']).str.strip()
        jobs_df['company'] = jobs_df['display_company'].apply(preprocess_text)

        # Add first word of company for fuzzy matching
        jobs_df['first_word_company'] = jobs_df['display_company'].str.split().str[0].apply(preprocess_text)

        # Precompute fuzzy matches for title and first_word_company
        jobs_df['merge_key'] = jobs_df['title'] + ' ' + jobs_df['first_word_company']
        merge_groups = {}
        visited_indices = set()

        for idx, row in jobs_df.iterrows():
            if idx in visited_indices:
                continue

            # Find close matches using rapidfuzz
            matches = rf_process.extract(
                row['merge_key'],
                jobs_df['merge_key'].tolist(),
                scorer=ratio,
                score_cutoff=80
            )

            # Collect indices of matching rows
            matched_indices = [match[2] for match in matches if match[2] != idx]
            visited_indices.update(matched_indices)

            # Add current index to group
            group_indices = [idx] + matched_indices
            merge_groups[idx] = group_indices

        # Consolidate rows based on merge groups
        consolidated_rows = []
        seen_indices = set()
        for group_idx, indices in merge_groups.items():
            if any(idx in seen_indices for idx in indices):
                continue

            # Mark all indices in the group as seen
            seen_indices.update(indices)

            # Combine apply options from all group members
            combined_apply_options = []
            for idx in indices:
                options = jobs_df.loc[idx, 'apply_options']
                if isinstance(options, list):
                    combined_apply_options.extend(options)

            # Remove duplicate options by apply_link
            combined_apply_links = list({clean_url(option['apply_link']) for option in combined_apply_options})

            # Consolidate row (keep the first row as base)
            base_row = jobs_df.loc[group_idx].copy()
            base_row['apply_links'] = combined_apply_links
            consolidated_rows.append(base_row)

        # Create new DataFrame from consolidated rows
        jobs_df = pd.DataFrame(consolidated_rows)

        # Drop the temporary columns
        jobs_df = jobs_df.drop(columns=['first_word_company', 'merge_key'], errors='ignore')

        # Generate the links column
        jobs_df['apply_options'] = jobs_df['apply_links'].apply(lambda links: [
            {
                'publisher': extract_domain(link),
                'apply_link': link
            }
            for link in links if
            not any(excluded in link.lower() for excluded in [s.lower() for s in Config.EXCLUDED_JOB_PUBLISHERS]) # exclude crappy job links
        ] if isinstance(links, list) else [])

        # Drop rows where `apply_options` is an empty list
        jobs_df = jobs_df[jobs_df['apply_options'].apply(len) > 0].reset_index(drop=True)

        # Drop intermediate columns
        jobs_df = jobs_df.drop(columns=['first_word_company', 'merge_key'], errors='ignore')

    return jobs_df






