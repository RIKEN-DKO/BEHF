import re
from django.shortcuts import render
from django.core.paginator import Paginator
import requests
import json

API_URL = "http://0.0.0.0:5555"

# Bio event mapping
BIO_EVENTS = {
    'cg': 'Cancer Genetics',
    'ge11': 'GENIA Event Extraction (GENIA), 2011',
    'ge13': 'GENIA Event Extraction (GENIA), 2013',
    'id': 'Infectious Diseases',
    'epi': 'Epigenetics and Post-translational Modifications',
    'pc': 'Pathway Curation',
    'mlee': 'Multi-Level Event Extraction',
}


def clean_string(s):
    return re.sub(r'[^a-zA-Z0-9]', 'C', s)


def find_non_alphanumeric(s):
    return set(re.findall(r'[^a-zA-Z0-9]', s))


def search(request):
    query = request.GET.get('query')
    event_type = request.GET.get('event_type')  # retrieve the selected event type
    alpha = float(request.GET.get('alpha', 0.1))  # retrieve the alpha parameter
    request.session['event_type'] = event_type
    if query is not None:
        response = requests.post(f"{API_URL}/search", json={
            "event_type": event_type,
            "query": query,
            'num_res': 50,
            'alpha': alpha  # pass the alpha parameter to your API
        })

        data = response.json()
        if data:
            # all_weird_chars = set()
            for result in data['results']:
                result['textAE']['text'] = result['textAE']['text'].replace(
                    "/", "0")

                result['textAE'] = json.dumps(result['textAE'])


            #     # Parse the string into a dictionary
            #     textAE_dict = json.loads(result['textAE'])

            #     # Access the 'text' key
            #     text = textAE_dict['text']

            #     # Find non-alphanumeric characters
            #     weird_chars = find_non_alphanumeric(text)
            #     # Update the global set with new characters
            #     all_weird_chars.update(weird_chars)

            # print("All Non-Alphanumeric Characters: ", all_weird_chars)




            paginator = Paginator(data['results'], 10)
            page_number = request.GET.get('page', 1)
            page_obj = paginator.get_page(page_number)
            return render(request, 'search_results.html', {
                    'query': query,
                    'results': page_obj,
                    'page_number': page_number,
                    'event_type': event_type,
                    'alpha': alpha,  # pass the alpha value to the template
                    'API_URL': API_URL
                })
        else:  
            return render(request, 'search.html', {
                'query': query,
                'message': "No results found for your query"
            })
    else:
        return render(request, 'search.html', {'alpha': 0.1})


import urllib.parse
import codecs

#...other code

def document(request, doc_id, resultTextAE):
    event_type = request.session.get('event_type', '')  # retrieve the selected event type from session
    response = requests.get(f"{API_URL}/annotations/{event_type}/{doc_id}")
    json_response = json.dumps(response.json())  # convert to JSON string with double quotes
    
    # Decode the URL-encoded string
    resultTextAE = urllib.parse.unquote(resultTextAE)
    
    # Decode escape sequences
    resultTextAE = codecs.decode(resultTextAE, 'unicode_escape')
    

    return render(request, 'document.html', 
                  {'textAE': json_response, 'doc_id': doc_id, 'resultTextAE': resultTextAE})
