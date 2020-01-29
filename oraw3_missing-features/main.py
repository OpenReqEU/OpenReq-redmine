import json
import os
import urllib.request

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/tracker')
def predict_tracker():
    title = request.args.get('t')
    body = request.args.get('b')

    if not title:
        return 'title query parameter is not provided', 400
    if not body:
        return 'body query parameter is not provided', 400

    params = json.dumps({
        'Title': title,
        'Body': body,
        'Stance': 0
    }).encode('utf8')

    req = urllib.request.Request(
        os.environ['BACKEND_RI'],
        method='POST',
        data=params,
        headers={'content-type': 'application/json'})

    resp = urllib.request.urlopen(req).read().decode('utf8')
    data = json.loads(resp)['data']

    if data['isAnomalyRatio'] > 0.5:
        req = urllib.request.Request(
            os.environ['BACKEND_SS'],
            method='POST',
            data=params,
            headers={'content-type': 'application/json'})

        resp = urllib.request.urlopen(req).read().decode('utf8')
        return jsonify({'topScoringIntent': {'intent': json.loads(resp)[0]}})

    return jsonify({'topScoringIntent': {'intent': 'Demand'}})

@app.route('/urgence')
def predict_urgence():
    title = request.args.get('t')
    body = request.args.get('b')

    if not title:
        return 'title query parameter is not provided', 400
    if not body:
        return 'body query parameter is not provided', 400

    params = json.dumps({
        'Title': title,
        'Body': body,
        'Stance': 0
    }).encode('utf8')

    req = urllib.request.Request(
        os.environ['BACKEND_SS'],
        method='POST',
        data=params,
        headers={'content-type': 'application/json'})

    resp = urllib.request.urlopen(req).read().decode('utf8')

    return jsonify({'topScoringIntent': {'intent': json.loads(resp)[1]}})