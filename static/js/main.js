        // Detect user's country using geoplugin.net
        // Location Detection (ipapi.co)
        function detectLocation() {
            fetch('https://ipapi.co/json/')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('location').value =  data.country_name;
                })
                .catch(() => {
                    document.getElementById('location').value = '';
                });
        }

        detectLocation();

        // Enable tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-mdb-toggle="tooltip"]'));
        const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new mdb.Tooltip(tooltipTriggerEl);
        });




        document.getElementById('searchButton').addEventListener('click', function(event) {
            event.preventDefault(); // Prevent the default form submission

            let valid = false;

            // Check which tab is active and validate the form accordingly
            if (document.getElementById('manualForm').checkValidity()) {
                valid = true;  // Form is valid
            } else {
                document.getElementById('manualForm').reportValidity();  // Show validation errors
            }


            if (valid) {
                // Show the loading section and hide the tabs & form
                document.getElementById('form-title').textContent = 'Results';
                document.getElementById('form-section').style.display = 'none';
                document.getElementById('loading-section').style.display = 'block';

                // Collect form data and send the AJAX request using Fetch API
                let formData;
                formData = new FormData(document.getElementById('manualForm'));


                fetch('/search-jobs', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Once data is received, hide the loading section and display results
                    document.getElementById('loading-section').style.display = 'none';
                    document.getElementById('results-section').style.display = 'block';

                    // Populate the results
                    let resultsContainer = document.getElementById('results-section');
                    resultsContainer.innerHTML = '';  // Clear any existing results

                    data.jobs.forEach(job => {
                        let jobCard = `
                            <a href="${job.job_url}" target="_blank" class="list-group-item list-group-item-action job-card">
                                <div class="d-flex justify-content-between align-items-start">
                                    <h5 class="mb-1">${job.title}</h5>
                                    <small>${job.date_posted}</small>
                                </div>
                                <p class="mb-1">${job.company}</p>
                                <small class="text-muted">Score: ${job.combined_score}</small>
                            </a>
                        `;
                        resultsContainer.innerHTML += jobCard;
                    });
                })
                .catch(error => {
                    // Handle errors appropriately
                    console.error('Error fetching job results:', error);
                });
            }
        });