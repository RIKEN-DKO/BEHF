
import pickle
import os
from pathlib import Path
import unidecode
import argparse
import tqdm 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dict_file", type=str, default='/home/julio/repos/pubmed_processed_data/dicts/pmid2info_small.pickle',
                        help="The dictionary file")

    parser.add_argument("--outdir", type=str, default='/home/julio/repos/event_finder/data/pubmed/',
                        help="the saving dir")

    return parser.parse_args()


def main(args):

    with open(args.dict_file,'rb') as handle:
        dic = pickle.load(handle)

    # Removing accents prevents  deepventmine to get stuck
    # accented_string = u'Poznań.'
    # # accented_string is of type 'unicode'
    # unaccented_string = unidecode.unidecode(accented_string)
    # # unaccented_string contains 'Malaga'and is of type 'str'
    # unaccented_string
    print('Creating four set of files.... ')
    chunk_len = len(dic) / 4

    #create dataset by divviding the data into 4 sets
    for i in range(1,5):
        Path(args.outdir+str(i) +'/text' ).mkdir(parents=True, exist_ok=True)

    setn = 1
    print('Creating set #',setn)
    for i, (pmid, values) in enumerate(tqdm.tqdm(dic.items())):

        with open(os.path.join(args.outdir+str(setn) + '/text', str(pmid)+'.txt'), 'w') as file:
            title = unidecode.unidecode(values[3])
            abstract = unidecode.unidecode(values[4])
            data = title + '\n' + abstract

            if len(data) > 20:
                file.write(data)  # 3 title, 4 abstract
        if i > chunk_len * setn:

            setn += 1
            print('Creating set #',setn)

    print('DONE ')

if __name__ == '__main__':
    main(parse_args())



