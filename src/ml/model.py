import joblib

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from pathlib import Path
import os


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model_obj, name):
    """Save the model to a .pkl file.

    Inputs:

    model_obj: The model to save

    name: The string name for the .pkl file
    """
    path = os.path.abspath(Path(__file__).parent / f"../../model/{name}.pkl")
    joblib.dump(model_obj, path)


def load_model(model):

    model_loaded = joblib.load(model)

    return model_loaded


def compute_performance_slices(df, model, cat_features, encoder, lb):

    all_metrics = []

    for feature in cat_features:
        for category in df[feature].unique():
            feature_slice = df.loc[df[feature] == category]

            # Reprocess data for slice inference

            X, y, _, _ = process_data(
                feature_slice,
                cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            predict = inference(model, X)

            precision, recall, fbeta = compute_model_metrics(y, predict)

            metric_data = f"{feature} = {category}  | precision: {precision}, \
                recall: {recall}, fbeta: {fbeta}"
            all_metrics.append(metric_data)

    export_slices_to_txt(all_metrics)


def export_slices_to_txt(data):
    path = os.path.abspath(Path(__file__).parent / "../slice_data/")
    with open(os.path.join(path, "slice_output.txt"), "w") as f:
        for line in data:
            f.write(f"{line}\n")
