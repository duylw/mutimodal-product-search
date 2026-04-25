import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Visual Search Engine", layout="wide")
st.title("🛍️ Multimodal Visual Search Engine")

tab1, tab2, tab3 = st.tabs(["Text Search", "Image Search", "Browse Database"])

def display_results(results):
    if not results:
        st.warning("No results found.")
        return
        
    cols = st.columns(len(results))
    for idx, col in enumerate(cols):
        item = results[idx]['item']
        score = results[idx]['score']
        with col:
            st.image(item['filepath'])
            st.markdown(f"**{item.get('name', 'Product')}**")
            st.caption(f"Similarity Score: {score:.3f}")

# Tab 1: Text Search
with tab1:
    query = st.text_input("Enter description:")
    if st.button("Search Text") and query:
        with st.spinner("Searching microservices..."):
            res = requests.post(f"{BACKEND_URL}/search/text", json={"query": query})
            display_results(res.json().get("results", []))

# Tab 2: Image Search
with tab2:
    uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and st.button("Search Image"):
        with st.spinner("Searching microservices..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            res = requests.post(f"{BACKEND_URL}/search/image", files=files)
            display_results(res.json().get("results", []))

# Tab 3: Browse Database (Pagination)
with tab3:
    st.subheader("All Items in Database")
    
    # Check total items
    try:
        res = requests.get(f"{BACKEND_URL}/items?skip=0&limit=1")
        total_items = res.json().get("total", 0)
    except Exception:
        total_items = 0
        
    if total_items > 0:
        limit_per_page = 3  # Set to a small number to easily test pagination with 5 items
        total_pages = (total_items + limit_per_page - 1) // limit_per_page
        
        # State for current page
        if 'page' not in st.session_state:
            st.session_state.page = 1
            
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.page > 1:
                st.session_state.page -= 1
                st.rerun()
        with col2:
            st.markdown(f"<div style='text-align: center'><b>Page {st.session_state.page} of {total_pages} (Total: {total_items} items)</b></div>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️") and st.session_state.page < total_pages:
                st.session_state.page += 1
                st.rerun()
                
        skip = (st.session_state.page - 1) * limit_per_page
        
        # Display items
        with st.spinner("Loading items..."):
            res = requests.get(f"{BACKEND_URL}/items?skip={skip}&limit={limit_per_page}")
            if res.status_code == 200:
                items = res.json().get("items", [])
                cols = st.columns(3)
                for idx, item in enumerate(items):
                    with cols[idx % 3]:
                        st.image(item['filepath'])
                        st.markdown(f"**{item['name']}**")
    else:
        st.info("No items in database yet or database is loading.")
