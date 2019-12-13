#!/bin/bash
set -e

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
#mkdir ./quora_data/train
#mkdir ./quora_data/test
#mkdir ./quora_data/dev

#mv quora_data/dev.tsv quora_data/dev/sents.tsv
#mv quora_data/train.tsv quora_data/train/sents.tsv
#mv quora_data/test.tsv quora_data/test/sents.tsv
python scripts/preprocess-quora.py

mkdir -p checkpoints/