import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

reference_data = pd.read_csv("../Data/drug200.csv")

current_data = reference_data.sample(
    n=100,
    random_state=42
).copy()

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)

snapshot.save_html("drift_report.html")

print("Data drift report generated successfully!")