import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Visual Search Engine", layout="wide")
st.title("🛍️ Multimodal Visual Search Engine (Microservices)")

tab1, tab2 = st.tabs(["Text Search", "Image Search"])

def display_results(results):
    if not results:
        st.warning("No results found.")
        return
        
    cols = st.columns(len(results))
    for idx, col in enumerate(cols):
        item = results[idx]['item']
        score = results[idx]['score']
        with col:
            # We mounted sample_images to the docker container so streamlit can access it
            st.image(item['filepath'])
            st.markdown(f"**{item.get('name', 'Product')}**")
            st.caption(f"Similarity Score: {score:.3f}")

with tab1:
    query = st.text_input("Enter description:")
    if st.button("Search Text") and query:
        with st.spinner("Searching microservices..."):
            res = requests.post(f"{BACKEND_URL}/search/text", json={"query": query})
            display_results(res.json().get("results", []))

with tab2:
    uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and st.button("Search Image"):
        with st.spinner("Searching microservices..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            res = requests.post(f"{BACKEND_URL}/search/image", files=files)
            display_results(res.json().get("results", []))
