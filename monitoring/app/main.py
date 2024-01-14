# Load data
import pickle
import datetime

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from http import HTTPStatus

from pathlib import Path

## monitoring
import pandas as pd
from evidently.report import Report

from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues


from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import make_asgi_app

app = FastAPI()

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Constants
model_filename = "model.pkl"
database_filename = Path("database.csv")

# Decalre the iris classes
classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]

# Load the saved model
with open(model_filename, "rb") as f:
    model = pickle.load(f)

## Create initial database.csv file
if not database_filename.exists():
    with open(database_filename, "w") as f:
        headers = "time, sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), prediction\n"
        f.write(headers)


@app.get("/")
def get_root():
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}


def write_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
):
    with open(database_filename, "a") as file:
        content = f"{ now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n"
        file.write(content)


@app.post("/iris_v1/")
async def iris_inference(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
) -> dict:
    data_points = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(data_points).item()

    background_tasks.add_task(
        write_database,
        str(datetime.datetime.now()),
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        pred,
    )
    return {"prediction": classes[pred], "predinction_int": pred}


@app.get("/empty_db/")
def empty_db(username: str, password: str):
    if username != "user" and password != "12345":
        return HTTPStatus.LOCKED
    with open(database_filename, "w") as f:
        headers = "time, sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), prediction\n"
        f.write(headers)


@app.get("/iris_monitoring/", response_class=HTMLResponse)
async def monitor_drift(username: str, password: str):
    if username != "user" and password != "12345":
        return HTTPStatus.LOCKED

    # get original dataset(s)

    iris = datasets.load_iris()

    reference_data = pd.DataFrame(iris.data, columns=iris.feature_names)

    database_filename = Path(__file__).parent.resolve()
    database_filename = database_filename / "database.csv"
    current_data = pd.read_csv(database_filename)

    current_data = current_data.iloc[:, [1, 2, 3, 4]]

    # run diagnostics
    report = Report(
        metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()]
    )
    report.run(
        reference_data=reference_data.iloc[:50],
        current_data=reference_data.iloc[50:101],
    )

    report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)

    # inst of saving, return html response(s)


Instrumentator().instrument(app).expose(app)
