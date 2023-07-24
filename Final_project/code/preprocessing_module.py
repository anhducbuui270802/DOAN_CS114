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
from underthesea import text_normalize


# load model phoBert và tokenizer của model đó
phoBert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)


# Xoá HTML tag
def remove_html(text):
  return re.sub(r'<[^>]*>', '', text)

# Đánh dấu spam URL
def check_url(text):
    pattern = re.compile(r'(http|https)://[^\s]+')
    match = pattern.search(text)
    if match:
        return True
    else:
        return False

# Chuẩn hóa Unicode và dấu câu
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

# Hàm tạo dict để convert định dạng cũ sang định dạng mới
def loaddicchar():
  dic = {}
  char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
  charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
  for i in range(len(char1252)):
      dic[char1252[i]] = charutf8[i]
  return dic

dicchar = loaddicchar()
def convert_unicode(text):
    text = re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], text)
    return text_normalize(text)

# Đưa về dạng viết thường
def to_lower_case(text):
    text = text.lower()
    return text

# Xoá các ký tự không cần thiết
def remove_unnecessary_charactor(text):
    # xóa các ký tự đặc biệt, emoji
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    # xóa kí tự chứa số
    text = re.sub(r'\w*\d\w*', '', text).strip()
    # xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Chuẩn hoá những từ lặp âm tiết
def remove_duplicate_characters(text):
    pattern = re.compile(r'(\w)\1{2,}')
    text = pattern.sub(r'\1', text)
    return text

# Chuẩn hoá viết tắt
def abbreviate(text, path = './abbreviations.txt'):
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
def word_token(text):
    return word_tokenize(text, format='text')

# xử lý stopword
def remove_stopwords(text, path = "./stopword.txt"):
    # stopwords = open(path)
    # stopwords = stopwords.readlines()
    # stopwords = [x.strip() for x in stopwords]
    stopwords = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            stopwords.append(line.strip())
    words = text.split(' ')
    res = list()
    for word in words:
        if word not in stopwords:
            res.append(word)

    return ' '.join(res)


# Feature extraction phoBERT
def feature_extraction(text):
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
    
    tokenized = [tokenize_line(text)]

    mask = [np.where(np.array(tokenized) == 1, 0, 1)]

    def extract_line(tokenized, mask):
        tokenized = torch.tensor(tokenized).to(torch.long)
        mask = torch.tensor(mask)

        with torch.no_grad():
            last_hidden_states = phoBert(input_ids=tokenized, attention_mask=mask)
        
        feature = last_hidden_states[0][:, 0, :].numpy()

        return feature

    return extract_line(tokenized, mask)


# Tách một đoạn văn thành nhiều câu
def extract_sectence_from_paragraph(paragraph):
    sentences = []
    for sentence in paragraph.split('.'):
        sentences.extend(sentence.split('\n'))
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
    return sentences
