from newspaper import Article, Config, ArticleException
from newspaper.utils import BeautifulSoup
import json
import sys
sys.path.append('..')
from data.train_bbc_data import preprocess_article_for_categorization
from data.train_cnn_data import preprocess_article_for_summarization

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
    
    original_text = article_text
    article_text = prepare_article(article_text)
    category = preprocess_article_for_categorization(article_text) # Get predicted category
    summary = preprocess_article_for_summarization(article_text) # Get predicted summary

    summarised_article = {'title': article.title, 'category': category, 'summary': summary, 'imageURL': article.top_image, 'article': original_text}
    return summarised_article

def prepare_article(article):
    """Prepares article for preprocessing by ensuring it is given as a single paragraph with lowercase characters."""

    article = article.replace('\r', '').replace('\n', '').replace(',', ' ').replace('–', ' ').replace('“', "").replace('\'','').replace('"', '').replace('”','').replace('\n\n', '')
    return article.lower()

def parse_bbc(article):
    """Parses BBC articles"""

    soup = BeautifulSoup(article.html, 'html.parser')
    
    content = soup.findAll("div", attrs={"data-component":"text-block"})

    text = ""
    for div in content:
        paragraphList = div.findAll('p')
        for paragraph in paragraphList:
            text += " " + paragraph.get_text() + "\n\n"

    return text