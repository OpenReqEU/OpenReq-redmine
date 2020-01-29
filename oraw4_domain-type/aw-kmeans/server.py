from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.route('/predict')
def predict():
    if not request.args.get('model_id') or not request.args.get('text'):
        return '"model_id" query parameter is required', 400

    print('loading model')
    vectorizer, model = pickle.load(open('model_%s.pkl' % request.args.get('model_id'), 'rb'))

    print('predicting')
    return jsonify(model.predict(vectorizer.transform([request.args.get('text')])).tolist()[0])

@app.route('/healthcheck')
def healthcheck():
    return 'ok', 200

print('ready')