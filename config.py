import os

class Config:
    UPLOAD_FOLDER = 'uploads/'
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'rtf'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'application/rtf'
    }
    CV_TEXT_LIMIT = 2000 #limit cv text to first 2000 words
    RESULTS_WANTED = 50
    DUMP_FILE_NAME = 'data/dump_search.csv'
    DEFAULT_RADIUS = 50
    INTERVAL_MAPPING = {'month': 30, 'week': 7, '3days': 3, 'today': 1}

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
