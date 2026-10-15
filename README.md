# TraceMind — Explainable Cyber Anomaly Detector

## About the Project

TraceMind is a beginner-friendly AI cybersecurity project that analyses login activity and identifies potentially suspicious behaviour.

It uses a machine learning model to detect unusual login patterns and provides simple explanations to help users understand why an event may be suspicious.

## Features

- Detects potentially unusual login activity using Isolation Forest.
- Explains suspicious events using understandable rules.
- Filters login records by username and detection result.
- Displays login activity in a chart.
- Shows Precision, Recall, F1-Score, and a Confusion Matrix for demonstration.
- Allows users to download login records as a CSV file.
- Generates synthetic login data for safe testing.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

## How It Works

1. The application generates synthetic login records.
2. The AI model analyses login hours and failed login attempts.
3. The model identifies records that appear unusual.
4. TraceMind displays the results and explanations in a dashboard.
5. Users can explore the records and download the data.

## How to Run

1. Install Python.
2. Install the required libraries:

   ```bash
   pip install streamlit pandas scikit-learn plotly
   ```

3. Open a terminal in the project folder.
4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Important Note

TraceMind is an educational prototype that uses synthetic data. Its evaluation labels are based on simple rules, so its performance scores do not establish real-world cybersecurity accuracy. It should not be used as a production security system.

## Future Improvements

- Test the model using a separate labelled dataset.
- Improve anomaly explanations.
- Add more login behaviour features.
- Compare different anomaly detection models.
- Improve the dashboard and reporting features.

## Author

H.G.Bulankulama.