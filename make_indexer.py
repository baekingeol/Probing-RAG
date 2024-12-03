# This script is modified from https://github.com/starsuzi/Adaptive-RAG/blob/main/retriever_server/build_index.py
import nltk
from nltk.tokenize import word_tokenize

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Document

nltk.download('punkt')
nltk.download('punkt_tab')

from typing import Dict
import json
import argparse

from typing import Any
import hashlib
import io
import dill
from tqdm import tqdm
import glob
import bz2
import base58
from bs4 import BeautifulSoup
import os
import random
import csv
# from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import SentenceTransformer
import faiss
#%%
def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_hotpotqa_documents(metadata: Dict = None):
    raw_glob_filepath = os.path.join("raw_data", "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")
    metadata = {"idx": 1}
    doc_id = 0
    documnets = []
    texts = []
    doc_ids = []
    # numbers = []
    # number =0
    assert "idx" in metadata
    for filepath in tqdm(glob.glob(raw_glob_filepath)):
        for datum in bz2.BZ2File(filepath).readlines():
            instance = json.loads(datum.strip())

            id_ = hash_object(instance)[:32]
            title = instance["title"]
            sentences_text = [e.strip() for e in instance["text"]]
            paragraph_text = " ".join(sentences_text)
            url = instance["url"]
            is_abstract = True
            paragraph_index = 0

            es_paragraph = {
                "id": id_,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph_text": paragraph_text,
                "url": url,
                "is_abstract": is_abstract,
                "_id": doc_id,
            }

            texts.append(paragraph_text)
            documnets.append(es_paragraph)

            doc_id += 1

    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    return return_texts, doc_ids, documnets

def make_iirc_documents():
    raw_filepath = os.path.join("raw_data", "iirc", "context_articles.json")
    metadata = {"idx": 1}
    doc_id = 0
    documnets = []
    texts = []
    doc_ids = []
    
    assert "idx" in metadata

    random.seed(13370)  # Don't change.

    with open(raw_filepath, "r") as file:
        full_data = json.load(file)

        for title, page_html in tqdm(full_data.items()):
            page_soup = BeautifulSoup(page_html, "html.parser")
            paragraph_texts = [
                text for text in page_soup.text.split("\n") if text.strip() and len(text.strip().split()) > 10
            ]

            paragraph_indices_and_texts = [
                (paragraph_index, paragraph_text) for paragraph_index, paragraph_text in enumerate(paragraph_texts)
            ]
            random.shuffle(paragraph_indices_and_texts)
            for paragraph_index, paragraph_text in paragraph_indices_and_texts:
                url = ""
                id_ = hash_object(title + paragraph_text)
                is_abstract = paragraph_index == 0
                es_paragraph = {
                    "id": id_,
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "paragraph_text": paragraph_text,
                    "url": url,
                    "is_abstract": is_abstract,
                    "id": metadata["idx"],
                }
                texts.append(paragraph_text)
                documnets.append(es_paragraph)
                doc_id += 1
                metadata["idx"] += 1
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
        
    return return_texts, doc_ids, documnets

def make_2wikimultihopqa_documents():
    raw_filepaths = [
        os.path.join("raw_data", "2wikimultihopqa", "train.json"),
        os.path.join("raw_data", "2wikimultihopqa", "dev.json"),
        os.path.join("raw_data", "2wikimultihopqa", "test.json"),
    ]
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    documnets = []
    texts = []
    doc_ids = []
    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):

                for paragraph in instance["context"]:

                    title = paragraph[0]
                    paragraph_text = " ".join(paragraph[1])
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                        "_id": metadata["idx"]
                    }

                    texts.append(paragraph_text)
                    documnets.append(es_paragraph)
                    doc_ids.append(doc_id)
                    doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids, documnets

def make_musique_documents():
    raw_filepaths = [
        os.path.join("raw_data", "musique", "musique_ans_v1.0_dev.jsonl"),
        os.path.join("raw_data", "musique", "musique_ans_v1.0_test.jsonl"),
        os.path.join("raw_data", "musique", "musique_ans_v1.0_train.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_dev.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_test.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_train.jsonl"),
    ]
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    documnets = []
    texts = []
    doc_ids = []
    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line)

                for paragraph in instance["paragraphs"]:

                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                        "_id": metadata["idx"],
                    }
                    # document = {
                    #     "_op_type": "create",
                    #     "_index": elasticsearch_index,
                    #     "_id": metadata["idx"],
                    #     "_source": es_paragraph,
                    # }
                    # yield (document)
                    texts.append(paragraph_text)
                    documnets.append(es_paragraph)
                    doc_ids.append(doc_id)
                    doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids, documnets

def make_wiki_documents():
    raw_glob_filepath = os.path.join("raw_data", "wiki", 'psgs_w100.tsv')
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    documnets = []
    texts = []
    doc_ids = []
    with open(raw_glob_filepath) as input_file:
        tr = csv.reader(input_file, delimiter='\t')
        next(tr)
        for line in tqdm(tr):
            #import pdb; pdb.set_trace()
            #dict_line['_id'] = line[0]
            paragraph_text = line[1]
            title = line[2]
            url = ""
            
            id_ = hash_object(" ".join([title, paragraph_text]))[:32]
            paragraph_index = 0
            is_abstract = True

            es_paragraph = {
                    "id": id_,
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "paragraph_text": paragraph_text,
                    "url": url,
                    "is_abstract": is_abstract,
                    "_id": metadata["idx"],
                                }
            
            texts.append(paragraph_text)
            documnets.append(es_paragraph)
            doc_ids.append(doc_id)
            doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids, documnets

def make_nq_documents():
    raw_filepaths = [
        # os.path.join("index", "nq", "biencoder-nq-train.json"),
        os.path.join("raw_data", "nq", "biencoder-nq-dev.json"),
    ]
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    texts, doc_ids = [], []
    
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):
                for value2 in instance['positive_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['hard_negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids  

def make_trivia_documents():
    raw_filepaths = [
        os.path.join("raw_data", "trivia", "biencoder-trivia-train.json"),
        os.path.join("raw_data", "trivia", "biencoder-trivia-dev.json"),
    ]
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    texts, doc_ids = [], []
    
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):
                for value2 in instance['positive_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['hard_negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids

def make_squad_documents():
    raw_filepaths = [
        os.path.join("raw_data", "squad", "biencoder-squad1-train.json"),
        os.path.join("raw_data", "squad", "biencoder-squad1-dev.json"),
    ]
    metadata = {"idx": 1}
    assert "idx" in metadata
    doc_id = 0
    texts, doc_ids = [], []
    
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):
                for value2 in instance['positive_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
                for value2 in instance['hard_negative_ctxs']:
                    doc_ids.append(doc_id)
                    texts.append(value2['text'])
                    doc_id += 1
    print('removing duplicates...')
    return_texts = list(set(texts))
    doc_ids = list(range(len(return_texts)))
    print('finish!!!')
    
    return return_texts, doc_ids      

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Index paragraphs in Elasticsearch")
    parser.add_argument(
        "--dataset_name",
        help="name of the dataset",
        type=str,
    )
    parser.add_argument(
        "--is_sparse",
        # help="force delete before creating new index.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args() #hotpotqa, 2wikimultihopqa, musique
    
    dataset_name = args.dataset_name
    
    if dataset_name == "hotpotqa":
        texts, doc_ids, documents = make_hotpotqa_documents()
    elif dataset_name == "iirc":
        texts, doc_ids, documents = make_iirc_documents()
    elif dataset_name == "2wikimultihopqa":
        texts, doc_ids, documents = make_2wikimultihopqa_documents()
    elif dataset_name == "musique":
        texts, doc_ids, documents = make_musique_documents()
    elif dataset_name == "wiki":
        texts, doc_ids, documents = make_wiki_documents()
    elif dataset_name == "trivia":
        texts, doc_ids = make_trivia_documents()
    elif dataset_name == "nq":
        texts, doc_ids = make_nq_documents()
    elif dataset_name == "squad":
        texts, doc_ids = make_squad_documents()
    else:
        raise Exception(f"Unknown dataset_name {dataset_name}")

    is_sparse = args.is_sparse
    model_retr_id = 'facebook/contriever-msmarco'
    if is_sparse:
        
        documents = []
        print('appending sparse index...')
        for num, text in tqdm(enumerate(texts)):
            documents.append(Document(text=text, doc_id=f'{num}'))
        
        docstore=SimpleDocumentStore()
        docstore.add_documents(documents)
        
        docstore.persist(f"raw_data/sparse_index/llama_index_bm25_model_{dataset_name}_2.json")

    elif is_sparse==False: 
        model = SentenceTransformer(model_retr_id)

        dimension = 768  
        index = faiss.IndexFlatL2(dimension)
        batch_size = 512
        steps = 0
        print('appending dense index...')
        for i in tqdm(range(0,len(texts), batch_size)):
            index.add(model.encode(texts[steps*batch_size : (steps+1)*batch_size]))
            steps += 1
        faiss.write_index(index, f"raw_data/dense_index/contriever_{dataset_name}_2.bin")

    import pandas as pd
    # import pdb;pdb.set_trace()
    df = pd.DataFrame([texts, doc_ids]).T
    df.columns = ['doc','doc_id']

    df.to_csv(f'raw_data/{dataset_name}_index_2.csv', index=False)
#%%
'''
python make_indexer.py --dataset_name hotpotqa --is_sparse
python make_indexer.py --dataset_name 2wikimultihopqa --is_sparse
python make_indexer.py --dataset_name musique --is_sparse
python make_indexer.py --dataset_name nq --is_sparse
python make_indexer.py --dataset_name trivia --is_sparse
'''    