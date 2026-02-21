import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import sys
sys.path.append(os.getcwd())
from src.data.scrape_data import determine_category, scrape_to_knowledge_base

class TestScraper(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "tests/temp_financial_docs"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_determine_category(self):
        """Test the category deduction logic."""
        # Mock article
        article = MagicMock()
        
        # Test 1: URL path deduction
        url1 = "https://www.investopedia.com/articles/basics/06/invest1000.asp"
        article.title = "Investing 101"
        self.assertEqual(determine_category(url1, article), "Investing")

        # Test 2: Keyword deduction (Retirement)
        url2 = "https://www.investopedia.com/terms/r/rothira.asp"
        article.title = "Roth IRA Rules"
        self.assertEqual(determine_category(url2, article), "Retirement")

        # Test 3: Keyword deduction (Taxes)
        url3 = "https://www.investopedia.com/terms/c/capitalgain.asp"
        article.title = "Capital Gains Tax"
        self.assertEqual(determine_category(url3, article), "Taxes")
        
        # Test 4: Default fallback
        url4 = "https://www.investopedia.com/random/article.asp"
        article.title = "Random Thing"
        self.assertEqual(determine_category(url4, article), "General_Finance")

    @patch('src.data.scrape_data.Article')
    def test_scrape_execution(self, MockArticle):
        """Test the scraping loop without actual network calls."""
        # Setup mock
        mock_instance = MockArticle.return_value
        mock_instance.title = "Test Article Title"
        mock_instance.text = "This is the content of the test article."
        mock_instance.download.return_value = None
        mock_instance.parse.return_value = None
        
        test_urls = ["https://www.investopedia.com/test/article.asp"]
        
        # Run function
        scrape_to_knowledge_base(test_urls, output_dir=self.test_dir)
        
        # Verify file creation
        # Logic says it defaults to "General_Finance" if no specific keywords match
        expected_path = os.path.join(self.test_dir, "General_Finance", "Test_Article_Title.txt")
        self.assertTrue(os.path.exists(expected_path))
        
        # Verify content
        with open(expected_path, "r") as f:
            content = f.read()
            self.assertIn("Title: Test Article Title", content)
            self.assertIn("This is the content of the test article.", content)

if __name__ == '__main__':
    unittest.main()
