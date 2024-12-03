mkdir -p raw_data/nq
cd raw_data/nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz

# Download TriviaQA
cd ..
mkdir -p trivia
cd trivia
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
gzip -d biencoder-trivia-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
gzip -d biencoder-trivia-train.json.gz

cd ..
mkdir -p wiki
cd wiki
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz