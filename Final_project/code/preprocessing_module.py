import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import math
import re
from underthesea import word_tokenize
from transformers import AutoModel, AutoTokenizer
import torch

# load model phoBert và tokenizer của model đó
phoBert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Xóa html tag
def remove_html(text):
  return re.sub(r'<[^>]*>', '', text)

# Đánh dấu spam URL (những cmt có URL thường là những cmt spam quảng cáo, có chèn các link)
def check_url(text):
    pattern = re.compile(r'(http|https)://[^\s]+')
    match = pattern.search(text)
    if match:
        return True
    else:
        return False
    
# Chuẩn hóa dấu câu
from underthesea import text_normalize
def convert_unicode(txt):
  return text_normalize(txt)

# Đưa văn bản về dạng viết thường
def to_lower_case(text):
    text = text.lower()
    return text

# Xóa các ký tự không cần thiết
def remove_unnecessary_charactor(document):
    # xóa các ký tự đặc biệt, emoji
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # xóa kí tự chứa số
    document = re.sub(r'\w*\d\w*', '', document).strip()
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document

# Chuẩn hóa những từ lặp âm tiết
def remove_duplicate_characters(text):
    pattern = re.compile(r'(\w)\1{2,}')
    text = pattern.sub(r'\1', text)
    return text


# Chuẩn hoá viết tắt
def abbreviate(text, path = './Final_project/code/abbreviations.txt'):
    # Đọc các cặp giá trị từ file văn bản
    replacements = {}
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            key, value = re.split(':', line, maxsplit=1)
            replacements[key] = value.strip()
    # Thay thế các giá trị trong chuỗi
    for key, value in replacements.items():
        text = re.sub(r'\b{}\b'.format(key), value, text)
    return(text)

# Tách từ tiếng Việt
def word_token(sentence):
    return word_tokenize(sentence, format='text')


# xử lý stopword
def remove_stopwords(document, path = "./Final_project/code/stopword.txt"):
    # stopwords = open(path)
    # stopwords = stopwords.readlines()
    # stopwords = [x.strip() for x in stopwords]
    stopwords = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            stopwords.append(line.strip())
    words = document.split(' ')
    res = list()
    for word in words:
        if word not in stopwords:
            res.append(word)

    return ' '.join(res)

# Tách một đoạn văn thành nhiều câu
def extract_sectence_from_paragraph(paragraph):
    sentences = []
    for sentence in paragraph.split('.'):
        sentences.extend(sentence.split('\n'))
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
    return sentences

# Feature extraction bằng phoBERT
def feature_extraction(document):
    # Đưa từng sentence qua tokenizer của PhoBERT để convert sang dạng token index với cùng chiều dài
    # params
    MAX_SEQ_LEN = 256 # chiều dài tối đa của một câu
    # id của 1 số token đặc biệt
    cls_id = 0  # đầu câu
    eos_id = 2  # cuối câu
    pad_id = 1  # padding

    # Hàm xử lý dữ liệu trên từng sentence
    def tokenize_line(line):
        tokenized = tokenizer.encode(line)
        
        l = len(tokenized)
        if l > MAX_SEQ_LEN: # nếu dài hơn thì cắt bỏ
            tokenized = tokenized[:MAX_SEQ_LEN]
            tokenized[-1] = eos_id # thêm EOS vào cuối câu
        else: # nếu ngắn hơn thì thêm padding vào
            tokenized = tokenized + [pad_id, ] * (MAX_SEQ_LEN - l)
        
        return tokenized
    
    tokenized = [tokenize_line(document)]

    mask = [np.where(np.array(tokenized) == 1, 0, 1)]

    def extract_line(tokenized, mask):
        tokenized = torch.tensor(tokenized).to(torch.long)
        mask = torch.tensor(mask)

        with torch.no_grad():
            last_hidden_states = phoBert(input_ids=tokenized, attention_mask=mask)
        
        feature = last_hidden_states[0][:, 0, :].numpy()

        return feature

    return extract_line(tokenized, mask)


