
import pickle
import os
from pathlib import Path
import unidecode
import argparse
import tqdm
import json
from bef.data.text_utils import preprocess_texts_DEM
import spacy 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dict_file", type=str, default='/home/julio/repos/pubmed_processed_data/query_engine/pmid2details.pickle',
                        help="The dictionary file")

    parser.add_argument("--outdir", type=str, default='/home/julio/repos/event_finder/data/pubmed/',
                        help="the saving dir")

    # parser.add_argument("--year", type=str, default='2018',
    #                     help='Extract data only from this year')

    return parser.parse_args()


def main(args):

    with open(args.dict_file, 'rb') as handle:
        dic = pickle.load(handle)

    docs_dict = dic
    # debug
    # for i,(pmid,values) in enumerate(dic.items()):
    #     docs_dict[pmid] = values
    #     if i >1000:
    #         break
    # Removing accents prevents  deepventmine to get stuck
    # accented_string = u'Poznań.'
    # # accented_string is of type 'unicode'
    # unaccented_string = unidecode.unidecode(accented_string)
    # # unaccented_string contains 'Malaga'and is of type 'str'

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # setn = 1
    # print('Creating set #',setn)
    # data_list = {}
    # for i, (pmid, values) in enumerate(tqdm.tqdm(dic.items())):

    #     title = unidecode.unidecode(values['title'])
    #     abstract = unidecode.unidecode(values['abstract'])
    #     data = title + '\n' + abstract 

    #     if len(data) > 20:
    #         data_list[pmid] = data  # 3 title, 4 abstract


    #     if i > 100:
    #         break
    nlp = spacy.blank("en")
    nlp.add_pipe('sentencizer')

    docs = docs_dict.values()
    docs = [unidecode.unidecode( value['title'] + value['abstract'] ) for value in docs]

    pmids = list(docs_dict.keys())
    
    print('preprocessing texts..')
    new_texts = preprocess_texts_DEM(docs, nlp,batch_size=1000)
    new_data = dict(zip(pmids,new_texts))
    
    print('saving..')
    with open(os.path.join(args.outdir , 'pubmed.json'), 'w') as file:
        json.dump(new_data,file)
        
if __name__ == '__main__':
    main(parse_args())
