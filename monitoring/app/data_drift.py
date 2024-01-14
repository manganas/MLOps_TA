import pandas as pd
from sklearn import datasets
from pathlib import Path

from evidently.report import Report

from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues


from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)


database_filename = Path.cwd() / "monitoring" / "app" / "database.csv"
report_filename = Path.cwd() / "monitoring" / "app" / "report.html"
test_filename = Path.cwd() / "monitoring" / "app" / "test_report.html"


iris = datasets.load_iris()

reference_data = pd.DataFrame(iris.data, columns=iris.feature_names)

# reference_data = datasets.load_iris(as_frame="auto").frame
current_data = pd.read_csv(database_filename)


# print(reference_data)
# print(current_data)

cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
current_data_df = current_data.iloc[:, [1, 2, 3, 4]]
# Standardize the dataframes such that they have the same column names
current_data_df.columns = cols

# print(current_data_df)


# report = Report(metrics=[DataDriftPreset()])
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data_df)
report.save_html(report_filename.as_posix())

##

data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data_df)
data_test.save_html(test_filename)
