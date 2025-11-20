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

from tokenizers import(
    tokenizer,
    decoders,
    pre_tokenizers,
    trainers,
    models,
    Tokenizer,
)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["<unk>", "<s>", "</s>"]

trainer = trainers.BpeTrainer(
    vocab_size = 256,
    speicial_tokens = special_tokens,
    show_progress = True,

)



