from newspaper import Article, Config

def fetch_article(url):
    """Fetches the article content"""

    article = Article(url)
    article.download()
    article.parse()

    print(article.text)
    summary = {'title': article.title, 'category': 'politics', 'summary': article.text, 'imageURL': article.top_image}
    
    return summary