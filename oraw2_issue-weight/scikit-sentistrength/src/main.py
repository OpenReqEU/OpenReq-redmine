from flask import Flask, jsonify, request

from classifier import run_ternary

app = Flask(__name__)

@app.route('/issue_weighting', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        query = content.get('Title') + ' ' + content.get('Body')
        stance = content.get('Stance')
    except Exception as e:
        return 'bad input or could not process.', 400

    return jsonify(run_ternary('oraw1_15k', query))

@app.route('/healthcheck')
def healthcheck():
    return 'ok', 200

print('ready')