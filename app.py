from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("kmeans_model.pkl")
pipeline = joblib.load("kmeans_preprocess_pipeline.pkl")

cluster_names = {
    0: "Loyal Regular",
    1: "Budget Conscious",
    2: "Occasional Buyer",
    3: "High Value",
    4: "At Risk"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    table = ""

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            try:
                if file.filename.endswith(".csv"):
                    df = pd.read_csv(file)
                elif file.filename.endswith(".xlsx"):
                    df = pd.read_excel(file)
                else:
                    prediction = "Please upload CSV or Excel file only."
                    return render_template("index.html", prediction=prediction, table=table)

                transformed = pipeline.transform(df)
                clusters = model.predict(transformed)

                df["Cluster_Number"] = clusters
                df["Customer_Segment"] = [
                    cluster_names.get(int(cluster), "Unknown Segment")
                    for cluster in clusters
                ]

                prediction = "Prediction completed successfully."
                table = df.head(10).to_html(index=False)

            except Exception as e:
                prediction = "Error: " + str(e)

    return render_template("index.html", prediction=prediction, table=table)

if __name__ == "__main__":
    app.run(debug=False)
