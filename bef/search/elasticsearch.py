import json
from elasticsearch import Elasticsearch


class ElasticSearchSeeker:
    def __init__(self,index_name = "pubmed_70s_cg",host="http://localhost:9200"):
        self.es = Elasticsearch(hosts=host)
        self.response = None
        self.index_name = index_name

    def retrieve_records(self,query):
        # Perform the search
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "name": query
                    }
                }
            }
        )
        self.response = response
        scores = []
        docids = []
        for hit in response["hits"]["hits"]:
            docids.append(hit["_id"])
            scores.append(hit["_score"])
        
        results = {}
        results['scores'] = scores
        results['records_ids'] = docids

        return results