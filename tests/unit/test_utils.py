import unittest
import pandas as pd
from io import BytesIO
from flask import Flask
from werkzeug.datastructures import FileStorage
from reportlab.pdfgen import canvas
from datetime import datetime
from src.utils import (
    preprocess_text, allowed_file, get_file_mime_type,
    extract_text_from_pdf, extract_text_from_docx, extract_text_from_rtf,
    validate_and_clean_input, process_job_dataframe
)

class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup a test Flask app
        cls.app = Flask(__name__)
        cls.app.config.update({
            "TESTING": True,
        })

        # Create mock PDF file for testing
        cls.sample_pdf = BytesIO()
        c = canvas.Canvas(cls.sample_pdf)
        c.drawString(100, 750, "Skilled Python developer 123")
        c.save()
        cls.sample_pdf.seek(0)  # Reset buffer position

        # Mock DOCX file
        cls.sample_docx = BytesIO(b'Sample DOCX Content')
        cls.sample_docx.seek(0)

    def _get_pdf_filestorage(self):
        # Create a new FileStorage object from the PDF BytesIO for each test
        self.sample_pdf.seek(0)  # Ensure buffer position is reset before reuse
        return FileStorage(
            stream=self.sample_pdf,
            filename="sample.pdf",
            content_type="application/pdf"
        )

    def test_allowed_file(self):
        # Test various file types
        self.assertTrue(allowed_file("resume.pdf"))
        self.assertTrue(allowed_file("resume.docx"))
        self.assertTrue(allowed_file("resume.txt"))
        self.assertFalse(allowed_file("resume.jpg"))  # Invalid file type

    def test_get_file_mime_type(self):
        # Create a temporary PDF file and check MIME type
        with open('sample.pdf', 'wb') as f:
            f.write(self.sample_pdf.getbuffer())
        self.assertEqual(get_file_mime_type('sample.pdf'), 'application/pdf')

    def test_extract_text_from_pdf(self):
        # Extract text from PDF and check content
        with open('sample.pdf', 'wb') as f:
            f.write(self.sample_pdf.getbuffer())
        text = extract_text_from_pdf('sample.pdf')
        self.assertIn("Skilled Python developer 123", text)

    def test_preprocess_text(self):
        # Test text preprocessing
        raw_text = "Hello World!! This is a test. 1_23"
        processed_text = preprocess_text(raw_text)
        self.assertEqual(processed_text, "hello world this is a test 1 23")

    def test_validate_and_clean_input(self):
        valid_form_data = {
            'search_terms': 'Software Developer, Python Engineer   ;  FULL-STACK-Developer ',
            'keywords': 'AI,,Machine-Learning,   deep_learning',
            'country': 'United States',
            'region': 'California',
            'posted_since': 'week'
        }

        # Test valid form with file
        file_storage = self._get_pdf_filestorage()

        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, {'cv_file': file_storage})
            self.assertIn('search_terms', result)
            self.assertEqual(len(result['search_terms']), 3)
            self.assertEqual(result['search_terms'], ['software developer', 'python engineer', 'full stack developer'])
            self.assertIn('cv_text', result)
            self.assertEqual(result['location'], 'California, United States')
            self.assertEqual(set(result['keywords']),
                             {'ai', 'deep', 'deep learning', 'developer', 'engineer', 'full',
                              'full stack developer', 'learning', 'machine', 'machine learning',
                              'python', 'python engineer', 'software', 'software developer', 'stack'})
            self.assertEqual(result['cv_text'], 'skilled python developer 123')


        # Test with country only
        file_storage = self._get_pdf_filestorage()
        valid_form_data['region'] = ''
        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, {'cv_file': file_storage})
            self.assertEqual(result['location'], 'United States')

        # Test missing search terms
        file_storage = self._get_pdf_filestorage()
        valid_form_data['search_terms'] = ''
        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, {'cv_file': file_storage})
            self.assertIsInstance(result, tuple)
            self.assertEqual(result[1], 400)

        # Test missing country
        file_storage = self._get_pdf_filestorage()
        valid_form_data['search_terms'] = 'Software Developer'
        valid_form_data['country'] = ''
        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, {'cv_file': file_storage})
            self.assertIsInstance(result, tuple)
            self.assertEqual(result[1], 400)

        # Test invalid file type
        invalid_file = FileStorage(stream=BytesIO(b"Some binary data"), filename="resume.jpg", content_type="image/jpeg")
        mock_files = {'cv_file': invalid_file}
        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, mock_files)
            self.assertIsInstance(result, tuple)
            self.assertEqual(result[1], 400)

        # Test invalid keywords
        valid_form_data['keywords'] = '!@#$%^&*()'
        valid_form_data['country'] = 'United States'
        with self.app.app_context():
            result = validate_and_clean_input(valid_form_data, {})
            self.assertEqual(len(result['keywords']), 3)  # Only search terms and it's words will be retained

    def test_process_job_dataframe(self):
        mock_jobs_data = pd.DataFrame({
            'title': ['Software Engineer', None, 'Data Scientist'],
            'company': ['Tech Corp', 'InnoWorks', None],
            'description': ['Develop software', 'Innovate solutions', 'Analyze data'],
            'date_posted': ['2022-10-01', None, '']
        })

        processed_df = process_job_dataframe(mock_jobs_data)
        self.assertIn('display_title', processed_df)
        self.assertIn('display_company', processed_df)

        today_str = datetime.today().strftime('%b %d')
        self.assertEqual(processed_df['date_posted'].iloc[1], today_str)
        self.assertEqual(processed_df['date_posted'].iloc[2], today_str)

        # test with empty description
        mock_data = pd.DataFrame({
            'title': ['Backend Developer'],
            'company': ['DevCo'],
            'description': [None],
            'date_posted': [None]
        })

        processed_df = process_job_dataframe(mock_data)
        self.assertEqual(processed_df['description'].iloc[0], 'backend developer')

        today_str = datetime.today().strftime('%b %d')
        self.assertEqual(processed_df['date_posted'].iloc[0], today_str)


if __name__ == '__main__':
    unittest.main()
