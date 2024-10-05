# Job Catcher

Job Catcher is a web application that allows users to rank job postings based on their CV text, using semantic and keyword matching. The application combines different approaches (SBERT, TF-IDF, and keyword scoring) to analyze job descriptions and match them with user profiles effectively.

## Features
- **Job Aggregator**: Fetches jobs from LinkedIn, Glassdoor, and Indeed.
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

5. **Install and Run Redis**:
    Redis is used to manage the rate limiting for API requests. Install Redis (on Linux):

      ```bash
      sudo apt update
      sudo apt install redis-server
      sudo systemctl start redis
      ```

    Verify Redis is running by typing:
    ```bash
    redis-cli ping
    ```
    If Redis is running, it should return `PONG`.

## Usage

### Starting the Application

1. **Run the Redis Server**:
    Redis must be running to handle rate limiting for API requests.
    ```bash
    redis-server
    ```

2. **Run the Flask Application**:
    You can start the application using one of the following fetchers:

    - Using the **scraper** fetcher:
      ```bash
      python app.py --fetcher scraper 
      ```

    - Using the **JSearch API** fetcher (more efficient, but requires API key):
      ```bash
      python app.py --fetcher jsearch 
      ```
   Note: You can also implement your own fetcher with other scrapers or API. Follow the code conventions in `src/fetchers/scraper.py`.

3. **Access the Application**:
   Open your web browser and go to `http://127.0.0.1:5001`.

### Providing Input
You can provide input directly in the web interface:
- **Search terms:** Enter comma-separated job titles (e.g., "product manager, financial advisor").
- **CV:** Upload your CV file for semantic comparison.
- **Skill-matching keywords:** Provide specific skills or keywords you want to match in job descriptions (e.g., "data analysis, project management").

### Viewing Results
After submitting the form, the application will display a ranked list of job postings based on their relevance to your CV. Each job listing includes the job title, company name, date posted, and a score indicating its relevance.

**Note:**
- A maximum of 50 results are displayed in the interface. The full results, including detailed scores, are saved to a CSV file located in the `data/` directory.
- The scores are normalized based on the job postings returned in the search.

## Project Structure
```
├── app.py
├── config.py
├── src/
│   ├── fetchers
│   │   ├── scraper.py
│   │   └──  scraper.py
│   ├── models.py
│   ├── ranking.py
│   └── utils.py
├── static/
│   ├── js
│   │   ├── crs.min.js
│   │   └── main.js
│   ├── avatar.png
│   └── styles.css
├── templates/
│   └──  index.html
├── data/
│   └── dump_search_scraper.csv (created at runtime)
├── uploads/ (created at runtime)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
## Configuration

- **Search Parameters:** Modify default the search parameters directly in `src/config.py`
- **Results Storage:** By default, results of the latest search are stored in a CSV file under the `data/` directory (refreshed after each search).
- **Customizing Matching Logic:** You are welcome to modify or replace the job-matching logic implemented in `src/models.py` and `src/ranking.py` to experiment with different matching strategies.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for review.

## License

This project is licensed under the terms of the GNU Affero General Public License (AGPL) v3.0.

### Commercial Use

For commercial use or licensing inquiries, please contact [ammar.halabi@gmail.com].

## Acknowledgements

This project uses the [JobSpy](https://github.com/Bunsly/JobSpy) package for job scraping. Also, sentence-transformers provides the core semantic-matching mechanism. Particularly, the mini models (e.g., [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)) provide a great balance between performance and efficiency.