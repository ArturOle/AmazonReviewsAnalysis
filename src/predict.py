from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# one-vs-one
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multioutput import MultiOutputClassifier

# one-vs-the-rest
from sklearn.neural_network import MLPClassifier

# multiclass-multioutput
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



import json
import pandas as pd
import numpy as np
# score from 0 to 5


data_train = []

with open("cell_phones_and_accessories.json", 'r') as file:
    data = json.load(file)

with open("prediction_cell_and_accessories.json", 'r') as file:
    data_pred = json.load(file)
i = 0

for key in data.keys():
    
    if key not in data_pred.keys():    
        if i > 8000:
            break
        i += 1
        j = 1
        for entry in data[key]:
            if j >1:
                break
            data_to_load = {}
            data_to_load["sentence"] = entry["title"] + entry["summary"] + entry["text"]
            data_to_load["score"] = float(entry["score"])
            data_train.append(pd.DataFrame.from_dict([data_to_load]))
            j+=1
            

df = pd.concat(data_train)
# df['label'] = df['label'].astype(np.uint8)
sentence = df["sentence"].values
score =    df["score"].values

sentence_train, sentence_test, y_train, y_test = train_test_split(
    sentence, 
    score, 
    test_size=0.8
)

vectorizer = CountVectorizer()
vectorizer.fit(sentence_train)

X_train = vectorizer.transform(sentence_train).toarray()
X_test  = vectorizer.transform(sentence_test).toarray()

nb = GaussianNB()
nb.fit(X_train, y_train)
score = nb.score(X_test, y_test)
print('Gaussian Naive Bayes accuracy: {}'.format(score))

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(X_train, y_train)
score = knc.score(X_test, y_test)
print('K-neighbors Classifier accuracy: {}'.format(score))

# svc = SVC(kernel="rbf", C=0.01)
# # svc.fit(X_train, y_train)
# # score = svc.score(X_test, y_test)

# print('Radial Basis Function kernel SVC accuracy: {}'.format(OneVsRestClassifier(svc).fit(X_train, y_train).score(X_test, y_test)))


# kernel = 1.0 * RBF(1.0)
# gcp = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)
# score = gcp.score(X_test, y_test)
# print('Gaussian Process Classifier accuracy: {}'.format(score))

mlp = OneVsRestClassifier(MLPClassifier(random_state=1, max_iter=10000)).fit(X_train, y_train)#.fit(X_train, y_train)
#score = mlp.score(X_test, y_test)
print('Multi-layer Perceptron Classifier accuracy: {}'.format(mlp.score(X_test, y_test)))

# forest = RandomForestClassifier(random_state=1)
# multi_target_forest = MultiOutputClassifier(forest, n_jobs=6)
# print(multi_target_forest.fit(X_train, y_train).score(X_test, y_test))

with open("prediction_cell_and_accessories.json", 'r') as file:
    data = json.load(file)
    
data_train = []
i = 0
for key in data.keys():
    for entry in data[key]:
        if i == 1000:
            break
        sentence_to_predict = pd.DataFrame.from_dict([entry])
        
        sentence_to_predict["sentence"] = sentence_to_predict["title"] + sentence_to_predict["summary"] + sentence_to_predict["text"]
        sentence = sentence_to_predict["sentence"].values
        data_to_predict = vectorizer.transform(sentence).toarray()

        entry["label"] = str(mlp.predict(data_to_predict)[0])
        # print(entry["label"])
        data_train.append(entry)
        i+=1

with open("predicted_test_score.json", "w+") as file:
    json.dump(data_train, file, indent=2)