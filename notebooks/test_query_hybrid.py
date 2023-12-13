# %%
%load_ext autoreload
%autoreload 2


# %%
from bef.search.bioevent_query_handler import BioEventData,BioEventQueryHandler,BioEventDataWithDB,MultiBioEventQueryHandler,BioEventQueryHandlerEL
from bef.search.elasticsearch import ElasticSearchSeeker
from bef.search.hybrid import HybridSeeker
from bef.search.bioevent_query_handler import MultiHybridBioEventQueryHandler

# %%
data_path = '/home/julio/repos/event_finder/data/pubmed_70s'
mhybrid_seeker =MultiHybridBioEventQueryHandler(['id','cg'],data_path=data_path)

# %%
mhybrid_seeker.search(event_type='id',query='disease',num_res=20,alpha=0.5)

# %%
mhybrid_seeker.get_doc_annotations(event_type='cg',doc_id='4460956')



# %%
