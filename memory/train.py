import os
import zipfile
import requests
import torch
import tqdm
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
    return spm.SentencePieceProcessor(f"{model_prefix}.model")

def parse_wikitext(text_path, encoding = "utf-8"):
    with open(text_path, "r", encoding = encoding) as f:
        text = f.read().split("\n \n \n")
    return text

def random_crop_to_length(array, length, n_special_tokens = 2):
    tensor_len = array.shape[0]
    diff = tensor_len - length
    start_idx = torch.randint(0, diff, (1,)).item()
    return array[start_idx:(start_idx + length - n_special_tokens)]


def collate2seqlen(batch, sp_model, recurrent_steps = 8, n_tokens = 256):
    min_seq_len = recurrent_steps * n_tokens
    batch = [sp_model.EncodeAsIds(line) for line in batch]
    batch = [random_crop_to_length(torch.tensor(line), min_seq_len, 2) for line in batch if len(line) > (min_seq_len - 2)]
    if batch:
        batch = torch.stack(batch)
        return batch

#TODO : maybe can remove short text or concat to desired length
class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text_path,
                 vocab_size = 8192,
                 model_type = "unigram",
                 model_prefix = "spm",
                 data_parse_function = parse_wikitext):
        self.text_path = text_path
        self.sp_model = generate_vocab(dir = self.text_path,
                                       vocab_size = vocab_size,
                                       model_type = model_type,
                                       model_prefix = model_prefix)
        self.data = data_parse_function(self.text_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dim = 512
    n_epochs = 10
    recurrent_steps = 8
    n_tokens = 256
    vocab_size = 8192
    bos_id = 1
    eos_id = 2
    memory_loss_weight = 0.01
    recall_loss_weight = 2
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    get_wikitext()
    wikitext = TextDataset("../data/wikitext-2-raw/wiki.train.raw")
    ars = ArsMemoria(dim = dim,
                       predictor_context = n_tokens,
                       embed_dim = vocab_size).to(device)
    optimizer = torch.optim.Adam(ars.parameters(), lr=1e-4)

    test_prompt = wikitext.sp_model.EncodeAsIds("Hello world! I am ")
    test_prompt = torch.tensor(test_prompt).unsqueeze(0).to(device)

    losses = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} / {n_epochs}")

        # example for this epoch
        completion = ars.sample(test_prompt.to(device), len = 10)
        print(wikitext.sp_model.DecodeIds(completion[0].tolist()))

        train_loader = torch.utils.data.DataLoader(wikitext,
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            collate_fn=lambda x: collate2seqlen(x, wikitext.sp_model))
        pbar = tqdm.tqdm(total=len(train_loader) * recurrent_steps)
        for batch in train_loader:
            if batch is not None:
                optimizer.zero_grad()
                memories = None
                # append eos and bos
                bos_tensor = torch.tensor([bos_id])
                eos_tensor = torch.tensor([eos_id])
                batch = torch.cat([bos_tensor.repeat(batch.shape[0], 1),
                                batch, 
                                eos_tensor.repeat(batch.shape[0], 1)], dim=1).to(device)

                for i in range(recurrent_steps):

                    x = batch[:, i*n_tokens:(i+1)*n_tokens]
                    embedded_labels = ars.embedder(x)
                    embed, memories, ar_loss = ars.autoregressive_loss(embedded_labels, 
                                                                    x,
                                                                    memories = memories)
                    ar_loss.backward()
                    # TODO : maybe make a detach + clone wrapper
                    recall_loss = ars.recall_loss(embedded_labels.detach().clone().requires_grad_(True), 
                                                memories.detach().clone().requires_grad_(True))
                    recall_loss = recall_loss * recall_loss_weight
                    recall_loss.backward()
                    optimizer.step()

                    memory_loss = ars.memory_loss(embed.detach().clone().requires_grad_(True), 
                                                memories.detach().clone().requires_grad_(True))
                    memory_loss = memory_loss * memory_loss_weight
                    memory_loss.backward()
                    optimizer.step()

                    memories = memories.detach().clone().requires_grad_(True)
                    pbar.update(1)
                    pbar.set_description(f"Losses : AR : {round(ar_loss.item())} | Recall : {round(recall_loss.item())} | Memory : {round(memory_loss.item())}")

                    losses.append(ar_loss.item() + recall_loss.item() + memory_loss.item())
        pbar.close()

