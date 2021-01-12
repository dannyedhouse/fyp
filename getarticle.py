from newspaper import Article, Config, ArticleException

def fetch_article(url):
    """Fetches the article content"""

    try:
        article = Article(url)
        article.download()
        article.parse()

    except ArticleException:
        return {}

    print(article.text)
    summary = {'title': article.title, 'category': 'politics', 'summary': article.text, 'imageURL': article.top_image}

    return summary