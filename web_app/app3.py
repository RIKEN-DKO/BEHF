import streamlit as st
import json
import requests


API = "http://0.0.0.0:5555"
def search(query):

    results = requests.post(API, json={
        "query": query,
        'num_docs': 20
    })
    results = json.loads(results.json())['results']
    return results


def search_engine():
    st.title("Search Engine")
    search_term = st.text_input("Enter a search term")
    results = []
    state = st.session_state.get("page", 0)
    updated_state = state
    if search_term:
        results = search(search_term)

    if results:
        st.write("Results:")
        per_page = 5
        total_pages = len(results) // per_page + (len(results) % per_page != 0)
        start = state * per_page
        end = min(start + per_page, len(results))
        for result in results[start:end]:
            st.write(result)

        if total_pages > 1:
            st.write("Page", state + 1, "of", total_pages)
            if state > 0:
                if st.button("Previous"):
                    updated_state = max(state - 1, 0)
            if state < total_pages - 1:
                if st.button("Next"):
                    updated_state = min(state + 1, total_pages - 1)
            st.session_state["page"] = updated_state
    else:
        st.write("No results found")


if __name__ == '__main__':
    search_engine()
