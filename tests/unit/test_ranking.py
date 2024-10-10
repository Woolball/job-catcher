import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from src.ranking import calculate_keyword_scores, calculate_tfidf_scores, calculate_sbert_scores, rank_job_descriptions

class TestRankingFunctions(unittest.TestCase):

    def test_calculate_keyword_scores(self):
        # Test with valid input
        keywords = ['python', 'developer']
        job_texts = ['Looking for a python developer', 'We need a Java developer']
        expected_scores = [1.0, 0.5]
        self.assertEqual(calculate_keyword_scores(keywords, job_texts), expected_scores)

        # Edge case: Empty keyword list
        self.assertEqual(calculate_keyword_scores([], job_texts), [0.0, 0.0])

        # Edge case: No matching keywords
        job_texts = ['We need a Java programmer']
        expected_scores = [0.0]
        self.assertEqual(calculate_keyword_scores(keywords, job_texts), expected_scores)

        # Test with all keywords matching
        job_texts = ['Looking for a python developer']
        expected_scores = [1.0]
        self.assertEqual(calculate_keyword_scores(keywords, job_texts), expected_scores)

    def test_calculate_tfidf_scores(self):
        # Test with valid input
        cv_text = 'python developer'
        job_texts = ['Looking for a python developer', 'We need a Java developer']
        result = calculate_tfidf_scores(cv_text, job_texts)
        self.assertEqual(len(result), 2)  # Expecting scores for two job descriptions

        # Edge case: Empty job descriptions
        job_texts = []
        result = calculate_tfidf_scores(cv_text, job_texts)
        self.assertEqual(len(result), 0)

        # Edge case: Empty CV text
        cv_text = ''
        job_texts = ['Looking for a python developer']
        result = calculate_tfidf_scores(cv_text, job_texts)
        self.assertEqual(len(result), 1)  # Should still return a score for one job description

        # Edge case: Identical CV and job description
        cv_text = 'python developer'
        job_texts = ['python developer']
        result = calculate_tfidf_scores(cv_text, job_texts)
        self.assertTrue(np.allclose(result, [1.0]))  # Expecting a perfect match score

    def test_calculate_sbert_scores(self):
        # Test with valid input
        cv_text = 'python developer'
        job_texts = ['Looking for a python developer', 'We need a Java developer']
        result = calculate_sbert_scores(cv_text, job_texts)
        self.assertEqual(len(result), 2)

        # Edge case: Empty job descriptions
        job_texts = []
        result = calculate_sbert_scores(cv_text, job_texts)
        self.assertEqual(len(result), 0)

        # Edge case: Empty CV text
        cv_text = ''
        job_texts = ['Looking for a python developer']
        result = calculate_sbert_scores(cv_text, job_texts)
        self.assertEqual(len(result), 1)

        # Edge case: Identical CV and job description
        cv_text = 'python developer'
        job_texts = ['python developer']
        result = calculate_sbert_scores(cv_text, job_texts)
        self.assertTrue(np.allclose(result, [1.0]))

    def test_rank_job_descriptions(self):
        # Test with valid input
        jobs_df = pd.DataFrame({
            'description': ['Looking for a python developer', 'We need a Java developer'],
            'title': ['Python Dev', 'Java Dev']
        })
        cv_text = 'python developer'
        keywords = ['python', 'developer']

        # Rank the job descriptions
        ranked_df = rank_job_descriptions(jobs_df, cv_text, keywords)

        # Check that the length of the returned DataFrame is the same as the input one
        self.assertEqual(len(ranked_df), len(jobs_df),
                         "The length of the ranked DataFrame should match the input DataFrame.")

        # 2. Ensure some values of each score type (tfidf, sbert_similarity, keyword_score) are non-zero
        self.assertTrue((ranked_df['tfidf_score'] != 0).any(), "Some TF-IDF scores should be non-zero")
        self.assertTrue((ranked_df['sbert_similarity'] != 0).any(), "Some SBERT similarity scores should be non-zero")
        self.assertTrue((ranked_df['keyword_score'] != 0).any(), "Some keyword scores should be non-zero")

        # 3. Ensure the rows are sorted in descending order by combined_score
        sorted_df = ranked_df.sort_values(by='combined_score', ascending=False).reset_index(drop=True)
        pd_testing.assert_frame_equal(ranked_df.reset_index(drop=True), sorted_df)

        # Edge case: Empty DataFrame
        empty_jobs_df = pd.DataFrame(columns=['description'])
        ranked_df = rank_job_descriptions(empty_jobs_df, cv_text, keywords)
        self.assertTrue(ranked_df.empty)

        # Edge case: Missing description field (we expect this to return an empty dataframe)
        jobs_df = pd.DataFrame({'title': ['Python Dev', 'Java Dev']})
        ranked_df = rank_job_descriptions(jobs_df, cv_text, keywords)
        pd.testing.assert_frame_equal(jobs_df, ranked_df)  # Expecting an empty DataFrame as description column is missing

if __name__ == '__main__':
    unittest.main()
