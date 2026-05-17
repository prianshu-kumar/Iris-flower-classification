import pickle
from pathlib import Path

import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"


@st.cache_data
def load_model(path: Path):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def main():
    st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
    st.title("Iris Flower Classification")
    st.write(
        "Use the saved KNN pipeline from `model/model.pkl` to predict the iris species from measurements."
    )

    st.sidebar.header("Input measurements")
    sepal_length = st.sidebar.number_input(
        "Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1
    )
    sepal_width = st.sidebar.number_input(
        "Sepal width (cm)", min_value=0.0, max_value=5.0, value=3.5, step=0.1
    )
    petal_length = st.sidebar.number_input(
        "Petal length (cm)", min_value=0.0, max_value=7.0, value=1.4, step=0.1
    )
    petal_width = st.sidebar.number_input(
        "Petal width (cm)", min_value=0.0, max_value=3.0, value=0.2, step=0.1
    )

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write("### Feature values")
    st.write(
        {
            "Sepal length": sepal_length,
            "Sepal width": sepal_width,
            "Petal length": petal_length,
            "Petal width": petal_width,
        }
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH)

    if st.button("Predict species"):
        prediction = model.predict(input_data)
        st.write("### Predicted species")
        st.success(prediction[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            classes = model.classes_
            st.write("### Prediction probabilities")
            st.table({classes[i]: f"{proba[i]:.2f}" for i in range(len(classes))})

    st.write("---")


if __name__ == "__main__":
    main()
