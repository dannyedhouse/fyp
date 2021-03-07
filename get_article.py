from newspaper import Article, Config, ArticleException
from newspaper.utils import BeautifulSoup
import json
import sys
sys.path.append('..')
from data.load_bbc_data import preprocess_article
from data.load_cnn_data import preprocess_article_summary

def fetch_article(url):
    """Fetches the article content"""

    try:
        article = Article(url)
        article.download()
        article.parse()

        if "bbc" in article.source_url:
            article_text = parse_bbc(article)
        else:
            article_text = article.text

    except ArticleException:
        return {}

    print(article_text)
    article_text = prepare_article(article_text)
    print("\n" + article_text)
    category = preprocess_article(article_text)
    summary_new = preprocess_article_summary(article_text)

    summary = {'title': article.title, 'category': category, 'summary': article_text, 'imageURL': article.top_image}
    return summary

def prepare_article(article):
    """Prepares article for preprocessing by ensuring it is given as a single paragraph with lowercase characters."""

    article = article.replace('\r', '').replace('\n', '').replace(',', ' ').replace('–', ' ').replace('“', "").replace('\'','').replace('"', '').replace('”','')
    return article.lower()

def parse_bbc(article):
    """Parses BBC articles"""

    soup = BeautifulSoup(article.html, 'html.parser')
    
    content = soup.findAll("div", attrs={"data-component":"text-block"})

    text = ""
    for div in content:
        paragraphList = div.findAll('p')
        for paragraph in paragraphList:
            text += " " + paragraph.get_text()

    return text