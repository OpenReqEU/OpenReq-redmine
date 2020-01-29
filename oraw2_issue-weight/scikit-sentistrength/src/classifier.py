import pickle
import sys

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer

from sentistrength import getSentiment
from load_dataset import load_dataset

pip = None

class SGDClassifier_(SGDClassifier):
    def __init__(self, loss='log'):
        super(SGDClassifier_, self).__init__(
            loss=loss,
            penalty="l2",
            class_weight="balanced",
            random_state=13)
    
    def predict_proba(self, X):
        return super(SGDClassifier_, self).predict_proba(X)

def train(datasetName):
    global pip
    datasetPath = '../data/' + datasetName
    modelPath = '../data/' + datasetName + 'Model.pkl'

    x, y = load_dataset(datasetPath)#pickle.load(open('tmp2.pkl', 'rb'))#
    pickle.dump((x, y), open('./tmp2.pkl', 'wb'))
    
    pip = Pipeline([
        ('vect', ColumnTransformer([
            ('tfidf', TfidfVectorizer(sublinear_tf=True), 0),
            ('senti', Normalizer(), [1])])),
        ('clf', MultiOutputClassifier(SGDClassifier_()))
    ])
    pip.fit(x, y)

    x, y = load_dataset(datasetPath, test=True)

    y_true_tracker, y_true_urgence = zip(*y)
    y_pred_tracker, y_pred_urgence = zip(*pip.predict(x))

    # prob = pip.predict_proba(x)
    # y_pred_tracker_seuil, y_pred_urgence = zip(*list(map(
    #     run_ternary, 
    #     zip(*prob))))
    
    print("tracker:")
    print(classification_report(y_true_tracker, y_pred_tracker))
    print("tracker seuil")
    # print(classification_report(y_true_tracker, y_pred_tracker_seuil))
    # print(confusion_matrix(y_true_tracker, y_pred_tracker_seuil))
    
    print("urgence:")
    print(classification_report(y_true_urgence, y_pred_urgence))

    pickle.dump(pip, open(modelPath, 'wb'))

def run(datasetName, x):
    global pip
    modelPath = '../data/' + datasetName + 'Model.pkl'

    if(not pip):
        try:
            pip = pickle.load(open(modelPath, 'rb'))
        except:
            train(datasetName)

    x = [x, *getSentiment(x)]

    return pip.predict_proba([x])

def run_ternary(datasetName, x):
    # [[Anomaly, Demand], [Basse, Haute, Normale]]
    x = run(datasetName, x)
    #print(x)
    tracker = x[0][0]
    anomaly_pb = tracker[0]
    urgence = x[1][0]

    result = ["", ""]
    if anomaly_pb > 0.8:
        result[0] = "Anomaly"
    elif anomaly_pb > 0.5:
        result[0] = "Human"
    else:
        result[0] = "Demand"
    
    if urgence[0] > urgence[1] and urgence [0] > urgence[2]:
        result[1] = "Basse"
    elif urgence[1] > urgence[2]:
        result[1] = "Haute"
    else:
        result[1] = "Normale"

    return result

if __name__ == '__main__':
    # print(run(sys.argv[1], sys.argv[2]))
    print(run_ternary(sys.argv[1], sys.argv[2]))