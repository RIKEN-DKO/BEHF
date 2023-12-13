import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from tqdm import tqdm


class SpladeBOW:

    def __init__(self) -> None:
        
        model_type_or_dir = "naver/splade-cocondenser-ensembledistil"
        model = Splade(model_type_or_dir, agg="max")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        self.reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

        # Check if CUDA is available and set the default tensor type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)


        self.model = model
        self.tokenizer = tokenizer

    def get_bow(self,doc): # now compute the document representation
        with torch.no_grad():
            doc_rep = self.model(d_kwargs=self.tokenizer(
                doc, return_tensors="pt",padding=True,truncation=True)
                )["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        # print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for k, v in sorted_d.items():
            bow_rep.append((self.reverse_voc[k], round(v, 2)))
        # print("SPLADE BOW rep:\n", bow_rep)
        return bow_rep
    
    def get_bows(self, docs, batch_size=64):
        bow_rep = []
        print('data -> model')
        num_batches = (len(docs) + batch_size - 1) // batch_size  # calculate total batches
        for i in tqdm(range(0, len(docs), batch_size), total=num_batches, desc='Processing Batches'):
            batch_docs = docs[i:i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_docs, return_tensors="pt",padding=True,truncation=True).to(self.device)
                batch_reps = self.model(d_kwargs=inputs)["d_rep"].squeeze()

            # print('model -> keywords')
            for doc_rep in batch_reps:
                curr_bow = []
                # Get the number of non-zero dimensions in the rep
                col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

                # Inspect the bow representation
                weights = doc_rep[col].cpu().tolist()
                d = {k: v for k, v in zip(col, weights)}
                sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

                for k, v in sorted_d.items():
                    curr_bow.append((self.reverse_voc[k], round(v, 2)))

                bow_rep.append(curr_bow)
        return bow_rep
