import streamlit as st
import requests
from PIL import Image
from io import BytesIO


backend_url = "http://127.0.0.1:8000/"


def main():
    st.title("Welcome")

    st.text(
        "You can see the pdf or kdeplot before and after applying our pipeline of a dataset from the following options"
    )

    model_choice = st.selectbox("Select a model", ("Iris", "Tips", "Titanic"))

    if st.button("Submit"):
        if model_choice == "Iris":
            body = {"dataset": "iris"}

        elif model_choice == "Tips":
            body = {"dataset": "tips"}

        elif model_choice == "Titanic":
            body = {"dataset": "titanic"}

        else:
            st.text("Will implement later")

        before_url = backend_url + "data-before"
        after_url = backend_url + "data-after"

        before_response = requests.post(before_url, json=body)
        before_fig = BytesIO(before_response.content)
        before_image = Image.open(before_fig)
        st.image(before_image)

        after_response = requests.post(after_url, json=body)
        after_fig = BytesIO(after_response.content)
        after_image = Image.open(after_fig)
        st.image(after_image)

        st.success("Done")


if __name__ == "__main__":
    main()
