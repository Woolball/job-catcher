from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pytest
from reportlab.pdfgen import canvas

@pytest.fixture
def sample_pdf_file(tmp_path):
    sample_pdf = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(sample_pdf))
    c.drawString(100, 750, "Valid PDF Content 123")
    c.save()
    return str(sample_pdf)

@pytest.fixture
def browser():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    yield driver
    driver.quit()



def test_job_search(browser, sample_pdf_file):
    browser.get("http://localhost:5001")  # Run Flask app on localhost

    # Fill in the form
    search_input = browser.find_element(By.ID, "search_terms")
    location_input = browser.find_element(By.ID, "location")
    keywords_input = browser.find_element(By.ID, "keywords")
    cv_file_input = browser.find_element(By.ID, "cv_file")

    search_input.send_keys("Data Scientist")
    location_input.send_keys("Zurich")
    keywords_input.send_keys("Python, Machine Learning")

    # Upload CV file
    cv_file_input.send_keys(sample_pdf_file)

    search_button = browser.find_element(By.ID, "searchButton")
    search_button.click()

    # Wait for results to load
    browser.implicitly_wait(10)  # Adjust wait time as needed

    job_results = browser.find_elements(By.CLASS_NAME, "job-card")
    assert len(job_results) > 0  # Ensure some jobs are displayed
