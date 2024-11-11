document.addEventListener('DOMContentLoaded', function() {
    const locationInput = document.getElementById('country');
    const jobSearchForm = document.getElementById('jobSearchForm');
    const searchButton = document.getElementById('searchButton');
    const searchButtonText = document.getElementById('searchButtonText');
    const searchSpinner = document.getElementById('searchSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const jobResults = document.getElementById('jobResults');
    const newSearchButton = document.getElementById('newSearchButton');
    const formTitle = document.getElementById('form-title');
    const navbar = document.querySelector('.navbar');

    // Correct selector for custom file upload div
    const fileUploadCustomDiv = document.querySelector('.custom-file-upload');
    const fileInput = document.getElementById('cv_file');
    const fileUploadLabel = document.querySelector('.file-upload-label');
    const fileUploadText = document.querySelector('.file-upload-text');

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

    // Detect user's location
    async function detectLocation() {
        try {
            const response = await fetch('https://ipapi.co/json/');
            const data = await response.json();
            locationInput.value = data.country_name;

            // Trigger change event after setting the country value
            const event = new Event('change');
            locationInput.dispatchEvent(event);

        } catch (error) {
            console.error('Error detecting location:', error);
            locationInput.value = '';
        }
    }

    detectLocation();

    // Custom validation on form submission
    jobSearchForm.addEventListener('submit', function(event) {
        const file = fileInput.files[0];
        if (file) {
            const allowedExtensions = /(\.pdf|\.docx|\.rtf|\.txt)$/i;
            if (!allowedExtensions.exec(file.name)) {
                event.preventDefault();
                event.stopImmediatePropagation();
                alert('Invalid file type. Please upload a .pdf, .docx, .rtf, or .txt file.');
                fileInput.value = ''; // Clear the input if invalid
                return false;
            }
        }
    });


    // Handle Job Search Form Submission
    jobSearchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        event.stopImmediatePropagation();
        searchButton.disabled = true;
        searchButtonText.textContent = 'Searching...';
        searchSpinner.style.display = 'inline-block';
        resultsSection.style.display = 'none';

        const formData = new FormData(jobSearchForm);

        try {
            const response = await fetch('/search-jobs', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok) {
                // Check if jobs are available
                if (data.jobs.length === 0) {
                    displayNoJobsMessage(data.message || "No jobs found. Please try different keywords.");
                } else {
                    displayResults(data.jobs);
                }
            } else {
                alert(data.error);
            }
        } catch (error) {
            console.error('Error fetching job results:', error);
            alert('An error occurred while searching for jobs. Please try again.');
        } finally {
            searchButton.disabled = false;
            searchButtonText.textContent = 'Search Jobs';
            searchSpinner.style.display = 'none';
        }
    });

        // Display No Jobs Found Message
        function displayNoJobsMessage(message) {
            jobResults.innerHTML = `
                <div class="alert alert-warning" role="alert">
                    ${message}
                </div>
            `;
            jobSearchForm.style.display = 'none';
            resultsSection.style.display = 'block';
        }


    // Display Job Search Results
    function displayResults(jobs) {
        jobResults.innerHTML = '';
        jobSearchForm.style.display = 'none';
        resultsSection.style.display = 'block';

        // Push state to browser history for results view
        history.pushState({ showResults: true }, '', '#results')

        jobs.forEach(job => {
            // Define color class based on the tier value
            let tierClass = '';
            switch (job.tier) {
                case 'High match':
                    tierClass = 'tier-high'; // Strong green for high
                    break;
                case 'Mid match':
                    tierClass = 'tier-moderate'; // Moderate green for medium
                    break;
                case 'Tiny maybe':
                    tierClass = 'tier-low'; // Faded green/yellow for low
                    break;
                case 'Irrelevant':
                    tierClass = 'tier-irrelevant'; // Grey for irrelevant
                    break;
            }
            const jobCard = `
                <a href="${job.job_url}" target="_blank" class="list-group-item list-group-item-action job-card">
                    <div class="d-flex justify-content-between align-items-start">
                        <h5 class="mb-1">${job.display_title}</h5>
                        <small>${job.date_posted}</small>
                    </div>
                    <p class="mb-1">${job.display_company}</p>
                    <small class="${tierClass}">
                        ${job.tier}
                    </small>
                </a>
            `;
            jobResults.innerHTML += jobCard;
        });
    }

    window.addEventListener('popstate', (event) => {
        if (event.state && event.state.showResults) {
            jobSearchForm.style.display = 'none';
            resultsSection.style.display = 'block';
        } else {
            jobSearchForm.style.display = 'block';
            resultsSection.style.display = 'none';
        }
    });


    // New Search Button
    newSearchButton.addEventListener('click', function() {
        jobSearchForm.style.display = 'block';
        resultsSection.style.display = 'none';
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Change navbar background on scroll
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
});

