

import requests
import streamlit as st



import json
import time
from typing import Dict
import pandas as pd


API_URL_CG = "http://0.0.0.0:5555"
API_URL_ID = "http://0.0.0.0:5556"


def print_results(results, max_results):

    # for i, _ in enumerate(results['records_ids']):
    for i, res in enumerate(results["results"]):

        eid = res['document']['event_id']
        score = res['total_score']
        nodes = res['document']['nodes']
        links = res['document']['links']
        pmid = res['document']['pmid']
        # name = node['name']
        title = '{}'.format(i+1)
        #Getting source and target nodes for link
        for link in links:
            for node in nodes:
                if link['source'] == node['id']:
                    source = node
                if link['target'] == node['id']:
                    target = node

            title += '<h5>[{}]<sub>{}</sub> ---<sub>[{}]</sub>--> [{}]<sub>{}</sub></h5>'.format(source['name'], source['type'],
            link['key'],
            target['name'],target['type'])

        st.write(title, unsafe_allow_html=True)
        # st.write('{} ID: {}'.format(int(id)))
        st.markdown('PMID: {} Event ID:{}, Score: {}'.format(pmid,eid,score))
        st.write('Graph Nodes')
        st.json(nodes,expanded=False)
        st.write('Graph Links')
        st.json(links,expanded=False)
        st.write('--------------------------------------------')


#############


def update_active_years(years_filter, seekers) -> None:

    for name, seeker in seekers.items():
        if name in years_filter or 'all' in years_filter:
            seeker._is_active = True
        else:
            seeker._is_active = False


def set_session_state():
    # set default values
    if 'search' not in st.session_state:
        st.session_state.search = None





def main():
    set_session_state()
    title = """<h2><span style="color: #d6eaf8;">{}</span></h2>"""
    # mseeker = create_mseeker()
    st.title("BEF: biomedical event finder")
    # query = st.text_input("Enter your query here:")

    if st.session_state.search is None:
        query = st.text_input('Enter your query:')
    else:
        query = st.text_input('Enter your query:', st.session_state.search)

    st.sidebar.markdown(title.format("Example queries:"), unsafe_allow_html=True)

    # st.sidebar.write("control of malignant tumor in childhood")
    st.sidebar.markdown("`Monocyte-derived macrophages`")
    st.sidebar.markdown("`melanoma`")
    st.sidebar.markdown("`stem cell sample`")
    st.sidebar.markdown("`tiger`")
    st.sidebar.write("")


    st.sidebar.markdown('__Available biomedical events:__')
    active_event = st.sidebar.selectbox(
    'Select type of event',['Cancer genetics','Infectious diseases'])


    MAX_DOCS_SHOW = 10
    if query:
        # col1.write("PMID")
        # col2.write("Link to article")
        # with st.spinner("Looking for info..."):
        start_time = time.time()
        # results = mseeker.retrieve_records(query, MAX_DOCS_SHOW)
        # Example EL.
        if active_event == 'Cancer genetics':
            API=API_URL_CG
        elif active_event =='Infectious diseases':
            API=API_URL_ID

        results = requests.post(API, json={
            "query": query,
            'num_docs': 5
        }).json()
        results = json.loads(results)
        # results = search_events(query, MAX_DOCS_SHOW)
        print('search took', time.time()-start_time)

        print_results(results,MAX_DOCS_SHOW)



if __name__ == '__main__':
    main()
