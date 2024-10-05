import pytest
from app import app
from reportlab.pdfgen import canvas

@pytest.fixture
def sample_pdf_file(tmp_path):
    sample_pdf = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(sample_pdf))
    c.drawString(100, 750, "Valid PDF Content 123")
    c.save()
    return str(sample_pdf)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Job Catcher" in response.data


def test_search_jobs(client, sample_pdf_file):
    data = {
        "search_terms": "Data Scientist",
        "location": "Zurich",
        "keywords": "Python, Machine Learning",
        "posted_since": "month"
    }

    with open(sample_pdf_file, 'rb') as cv_file:
        data['cv_file'] = (cv_file, 'sample.pdf')
        response = client.post('/search-jobs', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'jobs' in json_data
    assert len(json_data['jobs']) > 0


@pytest.fixture
def dummy_jpg_file(tmp_path):
    # Create a dummy .jpg file in the temporary directory
    invalid_file = tmp_path / "invalid_file.jpg"
    invalid_file.write_text("This is not a valid image file")  # Write some dummy text
    return invalid_file

def test_invalid_file_upload(client, dummy_jpg_file):
    data = {
        "search_terms": "Data Scientist",
        "location": "Zurich",
        "keywords": "Python, Machine Learning",
    }

    # Open the dummy .jpg file and upload it via the test client
    with open(dummy_jpg_file, 'rb') as file:
        data['cv_file'] = (file, 'invalid_file.jpg')
        response = client.post('/search-jobs', data=data, content_type='multipart/form-data')

    # Assert that the response status code indicates a bad request due to invalid file type
    assert response.status_code == 400
    assert b"Invalid file type or file too large" in response.data  # Adjusted to match the actual message

