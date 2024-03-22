from transformers import AutoTokenizer
from collections import defaultdict
"""
Byte-Pair encoding tokenizer 构建逻辑--主要包括两部分：“词频统计”+“词表合并”

输入:初始预料 + 分词器(word-base)

...... (waiting)
"""

def compute_pair_freqs(word_freqs, token2subtoken):
    """
    词频对统计
    Parameters
    ----------
    word_freqs: {'This': 3, 'Ġis': 2, 'Ġthe': 1, ....}
    splits: { Ġare:['Ġ', 'a', 'r', 'e'], .....}

    Returns
    -------

    """
    # 词频对统计字典
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        # 获取分词结果
        subtoken = token2subtoken[word]
        if len(subtoken) == 1:
            continue
        for i in range(len(subtoken) - 1):
            pair = (subtoken[i], subtoken[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(token1, token2, word_freqs, token2subtoken):
    """
    扩充词表
    Parameters
    ----------
    token1
    token2
    word_freqs
    token2num

    Returns
    -------

    """
    for word in word_freqs:
        tokenlist = token2subtoken[word]
        if len(tokenlist) == 1:
            continue

        i = 0
        while i < len(tokenlist) - 1:
            if tokenlist[i] == token1 and tokenlist[i + 1] == token2:
                tokenlist = tokenlist[:i] + [token1 + token2] + tokenlist[i + 2:]
            else:
                i += 1
            token2subtoken[word] = tokenlist

    return token2subtoken

def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])


if __name__ == "__main__":
    # 语料库
    print("----------------------------------------------准备语料库-------------------------------------------------")
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    # 加载分词器
    print("----------------------------------------------加载分词器-------------------------------------------------")
    tokenizer = AutoTokenizer.from_pretrained("/data/gzc/tokenizer/gpt2")
    print("----------------------------------------------词频率计算-------------------------------------------------")
    """
    GPT2TokenizerFast(name_or_path='/data/gzc/tokenizer/gpt2', 
    vocab_size=50257, model_max_length=1024, is_fast=True, padding_side='right', 
    truncation_side='right', 
    special_tokens={
        'bos_token': '<|endoftext|>', 
        'eos_token': '<|endoftext|>', 
        'unk_token': '<|endoftext|>'}, 
        clean_up_tokenization_spaces=True),  
        added_tokens_decoder={
	50256: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    """
    word_freqs = defaultdict(int) # 创建一个默认值为int类型的字典
    for text in corpus:
        # 对文本 pre-tokenize
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        # word is the token, offset是token的位置.
        print(words_with_offsets[0])
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    print(word_freqs)
    print("----------------------------------------------词汇表构建-------------------------------------------------")
    # 加入所有字符 并加入special token
    alphabet = list()

    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    # 基础词汇表构建
    vocab = ["<|endoftext|>"] + alphabet.copy()
    print(len(vocab))
    print("-----------------构建每个词到词表的映射token2subtoken, 并计算词表中任意两个词汇的频率-------------------------------------")
    token2subtoken = {word: [c for c in word] for word in word_freqs.keys()}

    # print(splits["This"])
    # 词频对统计
    pair_freqs = compute_pair_freqs(word_freqs, token2subtoken)

    for i, key in enumerate(pair_freqs.keys()):
        print(f"{key}: {pair_freqs[key]}")
        if i >= 5:
            break
    print("-----------------------------------------遍历出找出出现频率最高的词汇对---------------------------------------")
    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    print(best_pair, max_freq)
    print("---------------------------------------更新词表------------------------------------------------")
    merges = {best_pair: "".join(best_pair)}
    vocab.append("".join(best_pair))

    print(vocab)
    print("---------------------------------------更新tokens2num中的映射关系------------------------------------------------")
    token2subtoken = merge_pair(*best_pair, word_freqs, token2subtoken)
    print(token2subtoken)
    print("---------------------------------------预设词汇表大小，进行迭代------------------------------------------------")
    vocab_size = 100
    while len(vocab) < vocab_size:
        # 词对词频统计
        pair_freqs = compute_pair_freqs(word_freqs, token2subtoken)
        # 遍历出词频最高的词汇对
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

        token2subtoken = merge_pair(*best_pair, word_freqs, token2subtoken)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    print(vocab)
    print(merges)


