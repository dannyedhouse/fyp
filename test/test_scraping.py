import unittest
from get_article import fetch_article

class TestSources(unittest.TestCase):

    def test_guardian_source(self):
        url = "https://www.theguardian.com/world/2021/jan/14/regulator-refuses-to-approve-mass-covid-testing-schools-in-england"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")
    
    def test_bbc_source(self):
        url = "https://www.bbc.co.uk/news/uk-55681861"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")

    def test_metro_source(self):
        url = "https://metro.co.uk/2021/01/15/second-covid-wave-on-course-to-be-double-size-of-first-13911150/"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")

    def test_telegraph_source(self):
        url = "https://www.telegraph.co.uk/politics/2021/01/15/universal-credit-uplift-could-extended-plans-favoured-work-pensions/"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")

    def test_mirror_source(self):
        url = "https://www.mirror.co.uk/news/politics/covid-vaccine-expansion-five-million-23328923"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")

    def test_reuters_source(self):
        url = "https://www.reuters.com/article/us-usa-stocks/wall-street-closes-lower-as-banks-energy-shares-tumble-idUSKBN29K1BY"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")
    
    def test_cnn_source(self):
        url = "https://edition.cnn.com/2021/01/15/uk/bitcoin-trash-landfill-gbr-scli-intl/index.html"
        summary = fetch_article(url)
        summary_text = summary["summary"]
        self.assertNotEqual(summary_text, "")

if __name__ == "__main__":
    unittest.main()