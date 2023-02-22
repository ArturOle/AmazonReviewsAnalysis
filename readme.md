## Analysis of Amazon reviews on cell phones and accessories

<span style="color:gray">
    <tilted>
        Author: Artur Oleksiński
    </tilted>
</span>

#

Zip containing:

1) Input data txt to json (Julia) - transformation.ipynb
2) Sentiment analysis - sentiment_analysis.ipynb
3) Filtering if product is phone or accessories based on the review text - classification.ipynb
4) Predicting the score based on the review text - predict.py 

All the outputs are avaliable in the apropiate json_test files or notebooks

### >>> Input data txt to json (Julia)

Translation of txt files to structured json with product ID as key.

### >>> Sentiment analysis

Using Vader for sentiment analysis.

### >>> Filtering if product is phone or accessories based on the review text

Using scikit-learn, testing multiple machine learning algorithms.

Outcome: predicted_test_label.json

### >>> Predicting the score based on the review text

<span style="color:red">
    <strong>
        <h3 color="red"> Warning, memory heavy </h3>
    </strong>
</span>

| Method | Accuracy |
| ------- | -------- |
| Gaussian Naive Bayes | 0.338 |
| K-neighbors Classifier | 0.330 |
| Multi-layer Perceptron Classifier | 0.530 |

Outcome: predicted_test_score.json

Using scikit-learn, testing multiple machine learning algorithms and Vectorization methods.

