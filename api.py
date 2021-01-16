from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from get_article import fetch_article

app = Flask(__name__)
CORS(app)

# Endpoint for generating a summary
@app.route('/summary')
def fetch_summary():
    url = request.args.get('url')
    if url is None:
        abort(422)

    summary = fetch_article(url)
    return jsonify(summary)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)