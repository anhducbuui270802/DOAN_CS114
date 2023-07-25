import streamlit as st
import pandas as pd
import joblib
# import re
import numpy as np



# Import hàm từ phoBERT.ipynb
# import sys
# sys.path.insert(0, '../code')
# from preprocessing_module import remove_stopwords,\
#                                     word_token,abbreviate,remove_duplicate_characters,\
#                                     remove_unnecessary_charactor,to_lower_case,convert_unicode, \
#                                     remove_html, feature_extraction, extract_sectence_from_paragraph

import preprocessing_module as pre
from crawldata import crawl

def comment_preprocess(comment):
    # Xoá HTML tag
    comment = pre.remove_html(comment)

    # Chuẩn hoá dấu câu
    comment = pre.convert_unicode(comment)

    # Đưa về dạng viết thường
    comment =  pre.to_lower_case(comment)
    
    # Xoá các ký tự không cần thiết
    comment = pre.remove_unnecessary_charactor(comment)

    # Chuẩn hoá những từ lặp âm tiết
    comment = pre.remove_duplicate_characters(comment)
    
    # Chuẩn hoá viết tắt
    comment = pre.abbreviate(comment)

    # Tách từ tiếng Việt
    comment = pre.word_token(comment)

    # xử lý stopword
    comment = pre.remove_stopwords(comment)

    return comment



# Title
st.header("Phân tích cảm xúc bình luận")
svm = joblib.load("./Final_project/code/clf_logic.pkl")


# Phân tích cảm xúc trong bình luận
cmt_input = st.text_area('Comment here', '''''')

if cmt_input:

    # Đối với những cmt dài, gồm nhiều câu, tách đoạn cmt thành các câu ngắn để dự đoán chính xác hơn. 
    # Phân tích cảm xúc trên từng câu. 
    # Nếu số câu tích cực > Số câu tiêu cực, thì đoạn cmt đó sẽ là tích cực. Và ngược lại 
    # Nếu số lượng bằng nhau thì sẽ là trung lập
    cmts = pre.extract_sectence_from_paragraph(cmt_input)
    total = [0,0,0]
    # st.text(cmt)
    for cmt in cmts:
        cmt = comment_preprocess(cmt)
        feature = pre.feature_extraction(cmt)
        predict = svm.predict(feature)
        # st.text(cmt)
        if predict == -1:
            total[0] = total[0] + 1
            # st.text("😠 Tiêu cực")
        elif predict == 0: 
            total[1] = total[1] + 1 
            # st.text("😐 Trung lập")
        elif predict == 1:
            total[2] = total[2] + 1
            # st.text("😋 Tích cực")
    # st.text(total)
    if  total[2] - total[0] > 0 and total[2] - total[1] > 0:
        st.text("Kết luận: 😋 Bình luận tích cực")
    elif  total[0] - total[2] > 0 and total[0] - total[1] > 0:
        st.text("Kết luận: 😠 Bình luận tiêu cực")
    else:
        st.text("Kết luận: 😐 Bình luận trung lập")

# Tạo một form để nhập url
# with st.form(key='my_form'):
#     url_input = st.text_input(label="Enter the URL to the comment on foody 👇")

#     submitted = st.form_submit_button(label='Check')

#     if submitted:
#             try:
#                 with st.spinner('Wait for it...'):
#                     data = crawl(url_input)
#                     data["Preprocess_Review"] = data["Nội dung bình luận"].apply(lambda x: comment_preprocess(x))
#                     EMBED_SIZE = 768
#                     data_size = len(data)
#                     data_features = np.zeros(shape=(data_size, EMBED_SIZE))
#                     for index, row in data.iterrows():
#                         feature = pre.feature_extraction(row["Preprocess_Review"])
#                         data_features[index] = feature

#                     predicts = svm.predict(data_features)
#                     total = [0,0,0]
#                     for predict in predicts:
#                         if predict == -1:
#                             total[0] = total[0] + 1
#                         elif predict == 0:
#                             total[1] = total[1] + 1
#                         elif predict == 1:
#                             total[2] = total[2] + 1

#                     if  total[2] - total[0] > 0 and total[2] - total[1] > 0:
#                         st.text("Kết luận: 😋 Ăn quán này đi, được đấy!")
#                     elif  total[0] - total[2] > 0 and total[0] - total[1] > 0:
#                         st.text("Kết luận: 😠 Đừng ăn quán này, toàn chê")
#                     else:
#                         st.text("Kết luận: 😐 Sao cũng được ")
#                 st.success('Done!')
#             except:
#                 st.text("⚠️Link không hợp lệ⚠️")

    