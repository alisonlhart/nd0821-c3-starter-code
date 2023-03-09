# Script to train machine learning model.

import logging

from sklearn.model_selection import train_test_split
from src.ml.data import process_data, get_data
from src.ml.model import train_model, save_model, compute_performance_slices

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
def go():

    logger.info("Read data into dataframe")
    data = get_data()

    print(type(data))

    train, _ = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the train and test data
    logger.info("Process the train and test data")

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    logger.info("Train model on training data")
    model = train_model(X_train, y_train)

    logger.info("Save models")
    save_model(model, "model")
    save_model(encoder, "encoder")
    save_model(lb, "lb")

    logger.info("Compute metrics on data slices")
    compute_performance_slices(data, model, cat_features, encoder, lb)

    # Train and save a model.


if __name__ == "__main__":
    go()
