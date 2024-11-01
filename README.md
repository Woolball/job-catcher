<div style="float: left;">
   <img src="/static/favicon/favicon-48x48.png">
</div>
<h1>Job Catcher</h1>

Job Catcher is a web application that allows users to rank job postings based on their CV text. It is ultimately a retrieval and reranking pipeline. It first retrieves a pool of job postings, then ranks the results by combining several semantic and classical approaches to for an optimal match with the user profile.

**See the live website: https://job-catcher.onrender.com/**

## Features
- **Job Aggregator**: Uses Google’s job search API to gather listings from major job boards like LinkedIn, Glassdoor, Indeed, ZipRecruiter, Monster, and more, as well as from company career pages.
- **Advanced Matching**: Reranks the results using several techniques to optimize the match with the user profiel:
  - Semantic similarity with sentence-transformers.
  - Semantic elite keyword matching.
  - TF-IDF scoring.
  - Keyword scoring.
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
6. **Configure the Environment Variables**:
   Create a `.env` file in the root directory with the following content:

    ```
    JSEARCH_API_URL=https://jsearch.p.rapidapi.com/search
    JSEARCH_API_KEY=
    JSEARCH_API_HOST=jsearch.p.rapidapi.com
    JSEARCH_API_RATE_LIMIT_CALLS=5
    JSEARCH_API_RATE_LIMIT_PERIOD=1

    REDIS_HOST=localhost
    REDIS_PORT=6379

    FETCHER=scraper  # or 'jsearch' if you want to use the JSearch API
    ```

    Provide a value for `JSEARCH_API_KEY` with your actual JSearch API key if you want to use the jsearch fetcher. Using the **JSearch API** fetcher is more efficient, but [requires registration & API key](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch).


## Usage

### Starting the Application

1. **Run the Redis Server**:
    Redis must be running to handle rate limiting for API requests.
    ```bash
    redis-server
    ```

2. **Run the Flask Application**:
      ```bash
      python app.py 
      ```
3. **Access the Application**:
   Open your web browser and go to `http://127.0.0.1:5000`.

### Providing Input
You can provide input directly in the web interface:
- **Search terms:** Enter comma-separated job titles (e.g., "product manager, financial advisor").
- **CV:** Upload your CV file for semantic comparison.
- **Preferred and exclusion keywords:** Provide keywords you'd like to include or exclude from your results (e.g., "data analysis, project management, remote, senior").

### Viewing Results
After submitting the form, the application will display a ranked list of job postings based on their relevance to your CV. Each job listing includes the job title, company name, date posted, and a tag indicating its level of relevance.

**Note:**
- A maximum of 50 results are displayed by default in the interface. This can be configured in `/config.py`.
- After each search, the full results, including detailed scores, are saved to a CSV file located in the `data/` directory.

## Project Structure
```
├── app.py
├── config.py
├── src/
│   ├── fetchers
│   │   ├── scraper.py
│   │   └── jsearch.py
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
│   └── dump_search.csv (created at runtime)
├── uploads/ (created at runtime)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
## Configuration

- **App Parameters:** Modify the default parameters in `/config.py` and in your `/.env` file. Notably, you can select which of the two fetchers to use for retrieving job ads. You can also implement your own fetcher with other scrapers or API. Follow the code conventions in `src/fetchers/scraper.py`.
- **Results Storage:** By default, results of the latest search are stored in a CSV file under the `data/` directory (refreshed after each search).
- **Customizing Matching Logic:** You are welcome to modify or replace the job-matching logic implemented in `src/ranking.py` to experiment with different matching strategies.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for review.

## License

This project is licensed under the terms of the GNU Affero General Public License (AGPL) v3.0.

### Commercial Use

For commercial use or licensing inquiries, please contact [ammar.halabi@gmail.com].

## Acknowledgements

- This project uses the [**JobSpy**](https://github.com/Bunsly/JobSpy) package for job scraping.
- **SentenceTransformers** provides the core semantic-matching mechanism. Particularly, the mini models (e.g., [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) provides a great balance between performance and efficiency.
- **The Elite Keyword Matching** mechanism is inspired by: Susan, S., Sharma, M., & Choudhary, G. ([2024](https://doi.org/10.4114/intartif.vol27iss74pp117-132)). Uniqueness meets Semantics: A Novel Semantically Meaningful Bag-of-Words Approach for Matching Resumes to Job Profiles. Inteligencia Artificial, 27(74), 117–132.