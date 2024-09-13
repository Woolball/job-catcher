# Job Matcher

Job Search Matcher is a web application that allows users to rank job postings based on their CV text, using semantic and keyword matching. The application leverages powerful NLP models like SBERT and TF-IDF to analyze job descriptions and match them with user profiles effectively.

## Features
- **Job Search**: Fetches jobs from LinkedIn, Glassdoor, and Indeed.
- **Advanced Matching**: Combines semantic similarity and keyword matching to rank job postings based on relevance to the user's CV.
- **Web Interface**: Access via a simple web interface, with the local server running on the user's side.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Woolball/job-matcher.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd job-catcher
    ```

3. **Set Up a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install the Required Python Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Starting the Application

1. **Run the Flask Application**:
    ```bash
    python app.py
    ```

2. **Access the Application**:
   Open your web browser and go to `http://127.0.0.1:5000`.

### Providing Input

You can provide input directly in the web interface:
- **Search terms:** Enter comma-separated job titles (e.g., "product manager, financial advisor").
- **CV text:** Paste the textual content of your CV for semantic comparison.
- **Skill-matching keywords:** Provide specific skills or keywords you want to match in job descriptions (e.g., "data analysis, project management").

### Viewing Results

After submitting the form, the application will display a ranked list of job postings based on their relevance to your CV. Each job listing includes the job title, company name, date posted, and a score indicating its relevance.

**Note:**
- A maximum of 50 results are displayed in the interface. The full results, including detailed scores, are saved to a CSV file located in the `data/` directory.
- The scores are normalized based on the job postings returned in the search.

## Project Structure
```
├── app.py
├── __init__.py
├── templates/
│   ├── index.html
│   ├── results.html
├── static/
│   ├── js
│   │   └── main.js 
│   ├── styles.css
├── data/
│   └── dump_search_scraper.csv (created at runtime)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
## Configuration

- **Search Parameters:** Modify default job search parameters directly in `app.py`
- **Results Storage:** By default, results are stored in a CSV file under the `data/` directory.
- **Customizing Matching Models:** You are welcome to modify or replace the SBERT and TF-IDF models in app.py to experiment with different matching strategies.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for review.

## License

This project is licensed under the terms of the GNU Affero General Public License (AGPL) v3.0.

### Commercial Use

For commercial use or licensing inquiries, please contact [ammar.halabi@gmail.com].

## Acknowledgements

This project uses the [JobSpy](https://github.com/Bunsly/JobSpy) package for job scraping functionality. Special thanks to the JobSpy team for providing this helpful tool.