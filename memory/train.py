import os
import zipfile
import requests
import torch
import torchtext
import sentencepiece as spm
from ars_memoria import ArsMemoria

def get_wikitext(link = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
                 save_dir = "../data/"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(save_dir + "wikitext-2-raw"):
        r = requests.get(link, allow_redirects=True)
        open(save_dir + "wikitext-2-raw-v1.zip", "wb").write(r.content)
        
        with zipfile.ZipFile(save_dir + "wikitext-2-raw-v1.zip", 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        os.remove(save_dir + "wikitext-2-raw-v1.zip")

def generate_vocab(dir = "../data/wikitext-2-raw/wiki.train.raw", 
                   vocab_size = 8192, 
                   model_type = "unigram",
                   model_prefix = "spm"):
    if not os.path.exists(f"{model_prefix}.model"):
        spm.SentencePieceTrainer.Train(f"--input={dir} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}")

if __name__ == "__main__":
    get_wikitext()
    generate_vocab()


