from flask import Flask, jsonify, request
from flask_cors import CORS
from getarticle import fetch_article

app = Flask(__name__)
CORS(app)

# Endpoint for generating a summary
@app.route('/summary')
def hello_world():
    url = request.args.get('url')
    summary = fetch_article(url)
    return jsonify(summary)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)