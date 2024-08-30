# Job Matcher

Job Search Matcher is a web application that allows users to rank job postings based on their CV text, using semantic and keyword matching techniques. The application leverages powerful NLP models like SBERT and TF-IDF to analyze job descriptions and match them with user profiles effectively.

## Features
- **Flexible Input**: Users can upload their CV and search parameters as files or enter them directly via a web form.
- **Advanced Matching**: Combines semantic similarity and keyword matching to rank job postings based on relevance to the user's CV.
- **Caching System**: Efficient caching mechanism using SQLite to store and retrieve job search results, minimizing redundant searches and speeding up the process.
- **Responsive Design**: A user-friendly interface built with Bootstrap for seamless interaction on all devices.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Woolball/job-matcher.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd job-matcher
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

### Input Methods

You can provide input via either file upload or direct form entry:

- **File Upload**: see examples in the sample_input/ directory
  - Upload a `.txt` file containing your CV.
  - Upload a `.json` file with the search parameters (see the example below).

- **Form Entry**:
  - Paste your CV text and enter search parameters directly into the form fields.

### Example JSON Parameters File

Here's an example of a JSON file you can upload:

```json
{
    "search_terms": "Product Manager, Project Manager, Digital Product Manager, Product Owner, Product Strategy, Business Analyst, Technology Consultant, Product Innovation, Agile Product Manager, Product Development",
    "keywords": "Product Management, Strategic Planning, Roadmap, Agile, Cross-functional, Team Leadership, Market Research, Competitive Analysis, Stakeholder Management, Financial Forecasting, Data Analysis, User Research, A/B Testing, Product Analytics, Go-to-Market Strategy, Feature Prioritization, Business Growth, Customer Satisfaction, Revenue Growth, Digital Transformation",
    "time_window": 30,
    "location": "United States",
    "distance": 50
}
```

### Viewing Results

After submitting the form or files, the application will display a ranked list of job postings based on their relevance to your CV. Each job listing includes the job title, company name, date posted, and a score indicating its relevance.

## Project Structure

├── app.py
├── __init__.py
├── templates/
│   ├── index.html
│   ├── results.html
├── static/
│   ├── styles.css
├── data/
│   └── job_search_cache.db (generated at runtime)
├── uploads/ (uploaded at runtime)
├── sample_input/
│   ├── sample_cv.txt
│   ├── sample_params.json
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

## Configuration

- **Search Terms & Keywords:** Search terms and keywords are entered by the user.
- **Search Terms & Keywords:** Other search parameters can be modified directly in the app.py script.
- **Database:** The SQLite database is used for caching search results and is stored in the data/ directory.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for review.

## License

This project is licensed under the terms of the GNU Affero General Public License (AGPL) v3.0. 

### Commercial Use

For commercial use or if you require a different licensing arrangement, please contact [ammar.halabi@gmail.com].
