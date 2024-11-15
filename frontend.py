import streamlit as st
import requests
from PIL import Image
from io import BytesIO


backend_url = "http://127.0.0.1:8000/"


def main():
    st.title("Welcome")

    if st.button("Press to see before and after of Iris dataset"):
        before_url = backend_url + "iris-before"

        before_response = requests.get(before_url)

        before_fig = BytesIO(before_response.content)

        before_image = Image.open(before_fig)
        st.image(before_image)

        after_url = backend_url + "iris-after"

        after_response = requests.get(after_url)

        after_fig = BytesIO(after_response.content)

        after_image = Image.open(after_fig)
        st.image(after_image)

        st.success("Done")


if __name__ == "__main__":
    main()
