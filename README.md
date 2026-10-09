# Sentinel: a neural network watching card transactions

A small neural network that flags fraudulent credit card transactions, with a live demo page.

**Live demo:** https://credit-card-detection-4.onrender.com/
**Training notebook:** https://www.kaggle.com/code/goodnessabel/notebooka40eb32fc9/edit

## What it does

The demo feeds real transactions from a held out test set to the network, one after another. For each one you see the network's fraud probability, whether it was flagged, and then the ground truth: caught, missed, false alarm, or correct. A tally keeps score.

## The model

- Data: the public Kaggle credit card fraud dataset (European card transactions, 2013). Each transaction has Time, Amount, and 28 anonymised PCA components (V1 to V28).
- Network: a feed forward network in Keras, layers of 64, 32 and 16 units with ReLU, and one sigmoid output.
- Preprocessing: Time and Amount are standardised with a scaler fitted on the training set only.
- Split: train, validation (cv) and test, stratified so each has the same share of fraud.
- Threshold: 0.1, chosen on the validation set to favour catching fraud over avoiding false alarms.

## Results

| Model | PR-AUC (validation) |
|---|---|
| Neural network | 0.787 |
| Logistic regression | 0.679 |
| Gradient boosting (default settings) | 0.279 |

On the unseen test set at threshold 0.1, the network reached recall 0.857 and precision 0.677. The test set holds about 98 frauds, so these figures can move by several points. Roughly one in three alerts is a false alarm.

## Limits

- One public dataset from 2013, with very few fraud cases.
- Gradient boosting was only tried with default settings, so the comparison is "best of the models tested", not "best possible model".
- The demo stream contains far more fraud than real traffic, so the on page tallies are not real world rates.
- This is a learning project. Do not use it for real payment decisions.

## How it is deployed

The model was trained in Kaggle. `export_model.py` copies its weights into `artifacts/`, and `app.py` (Flask) runs the same forward pass in plain numpy, so the web service does not need TensorFlow. The page in `static/index.html` draws the network live.

```
app.py              web server
static/index.html   interface
artifacts/          exported weights and demo transactions
export_model.py     run in the notebook to create artifacts/
requirements.txt    dependencies
render.yaml         Render deploy settings
```


