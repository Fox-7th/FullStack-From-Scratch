# 思路
"""
选择tokenizer type
选择 pre_tokenizer
选择 decoder
设定special tokens
选择 tokenizer trainer，设定参数

训练tokenizer的数据准备 格式
制造一个tokenizer的 generator（iterator）
用generator训练tokenizer trainier

检查 训练过后的tokenizer，special_token之类
保存tokenizer

设定tokenizer的config（配置文件），保存；
有意思的是 配置文件，是保存tokenizer之后设定的

载入 tokenizer
尝试 用 配置文件 里的模版，输入一条sample
，看看能不能用

观察一下tokenizer的一些参数
观察分词结果 token_id

观察token_id 转化会text 的结果

"""

"""
# Hugging Face Tokenizers 库。
常用于 自行训练 BPE、WordPiece、Unigram 等模型的词表
pip install tokenizers

"""
# Repositories
# from ast import ExtSlice
from transformers import AutoTokenizer
import json
import os


# Create directories and paths
data_dir = "data"
train_file_name = "tokenizer_data.jsonl"
train_data_path = os.path.join(data_dir, train_file_name)
os.makedirs(data_dir, exist_ok = True)

tokenizer_dir = "model"
tokenizer_name = "tokenizer.json"
tokenizer_path = os.path.join(tokenizer_dir, tokenizer_name)
os.makedirs(tokenizer_dir, exist_ok = True)

config_name = "tokenizer_config.json"
tokenizer_config_path = os.path.join(tokenizer_dir, config_name)

print(f"Diretories Created.")


# store tokenizer config
tokenizer_config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "1": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "2": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    },
    "additional_special_tokens": [],
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<unk>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<unk>",
    "chat_template": "{{ '<s>' + messages[0]['text'] + '</s>' }}"
}

with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(tokenizer_config, config_file, ensure_ascii=False, indent=4)

print("Tokenizr Config Saved!")


# load tokenizer tools
from tokenizers import(
    tokenizer,
    decoders,
    pre_tokenizers,
    trainers,
    models,
    Tokenizer,
)
# settings
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["<unk>", "<s>", "</s>"]

trainer = trainers.BpeTrainer(
    vocab_size = 256,
    speicial_tokens = special_tokens,
    show_progress = True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
)

# load train data
def load_data(data_path):
    with open(data_path, "r", encoding = 'utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]
    
train_data_iter = load_data(train_data_path)
print(f"First Data: {next(train_data_iter)}")

# train
tokenizer.train_from_iterator(train_data_iter, trainer = trainer)
print(f"Data Loaded & Model Trained!")

# save
tokenizer.save(tokenizer_path)
tokenizer.model.save(tokenizer_path)

tokenizer_loaded = AutoTokenizer.from_pretrained(tokenizer_dir)
print(f"Model saved and loaded")

# Test
# txt -> templated txt
msg = [{"text": "我们认为下列真理不言而喻 : 人人生而平等，造物者赋予其若干不可剥夺的权利，包括生命权、自由权和追求幸福的权利"}]
msg_templated = tokenizer.apply_chat_template(
    msg,
    tokenize = False
)
print(f"Original Input:{msg}")
print(f"Templated Input:{msg_templated}")
print(f"Vocab Size:{tokenizer.vocab_size}")

# templated Txt -> input ids
msg_tokenized = tokenizer(msg_templated)
print(f"Original Input:{msg}")
print(f"Tokenized Input:{msg_tokenized}")

# Input_ids -> txt
id2txt_skip = tokenizer.decode(msg_tokenized["input_ids"], skip_special_tokens = True)
id2txt = tokenizer.decode(msg_tokenized["input_ids"], skip_special_tokens = False)

print(f"Decoded all token ids: {id2txt}")
print(f"Decoded non_special token ids: {id2txt_skip}")






