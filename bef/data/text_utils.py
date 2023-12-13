import regex


def preprocess_text_DEM(text,nlp):
    
    sentences = [sent.text for sent in nlp(text).sents]
    print(sentences)
    sentences = [add_spaces(sent) for sent in sentences]
    return sentences

#Batch spacy
def preprocess_texts_DEM(texts, nlp,batch_size=100):

    # Default batch size is `nlp.batch_size` (typically 1000)
    # docs = nlp.pipe(texts, n_process=2, batch_size=2000)
    docs = nlp.pipe(texts, batch_size=batch_size)
    new_texts =[]
    for doc in docs:
        sentences = [sent.text for sent in doc.sents]
        sentences = [add_spaces(sent) for sent in sentences]
        new_texts.append(sentences)

    return new_texts

def add_spaces(text):
    new = regex.sub(
        r"\s", " ", text
    )
    new = regex.sub(r'([^\w\s])', r' \1 ', new)

    return new


