import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def read_files(path):
    file_contents = []
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='GBK',errors='ignore') as f:
                file_contents.append(f.read())
                file_names.append(file)
    return file_names, file_contents

def preprocess_text(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='GBK',errors='ignore') as f:
        return [line.strip() for line in f.readlines()]

def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
    return vectorizer.fit_transform(docs)

def rank_documents(query_vec, docs_tfidf_matrix):
    cosine_similarities = cosine_similarity(query_vec.reshape(1, -1), docs_tfidf_matrix)
    return np.argsort(cosine_similarities, axis=-1)[:, ::-1]

def search_query(query: str, train_file_names: List[str], train_docs: List[str], stopwords: List[str], vectorizer: TfidfVectorizer, train_tfidf_matrix) -> List[str]:
    preprocessed_query = preprocess_text(query, stopwords)
    query_vec = vectorizer.transform([preprocessed_query])
    ranked_docs = rank_documents(query_vec, train_tfidf_matrix)
    return [train_file_names[i] for i in ranked_docs[0]]

if __name__ == '__main__':
    train_set_path = '训练集'
    test_set_path = '测试集'
    stopwords_file = 'stopwords.txt'

    # 读取训练集和测试集
    train_file_names, train_docs = read_files(train_set_path)
    test_file_names, test_docs = read_files(test_set_path)

    # 加载停用词表
    stopwords = load_stopwords(stopwords_file)

    # 预处理文档
    train_docs = [preprocess_text(doc, stopwords) for doc in train_docs]
    test_docs = [preprocess_text(doc, stopwords) for doc in test_docs]

    # 创建TF-IDF向量矩阵
    tfidf_matrix = create_tfidf_matrix(train_docs + test_docs)
    train_tfidf_matrix = tfidf_matrix[:len(train_docs)].toarray() # type: ignore
    test_tfidf_matrix = tfidf_matrix[len(train_docs):].toarray() # type: ignore
    
    # 打印矩阵形状
    # print("训练集TF-IDF矩阵形状:", train_tfidf_matrix.shape)
    # print("测试集TF-IDF矩阵形状:", test_tfidf_matrix.shape)

    # 创建TF-IDF向量矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(train_docs + test_docs)
    train_tfidf_matrix = tfidf_matrix[:len(train_docs)] # type: ignore

    # 查询
    query = "NBA：湖人主帅杰克逊支持科比"
    search_result = search_query(query, train_file_names, train_docs, stopwords, vectorizer, train_tfidf_matrix)
    result_str = ', '.join(search_result)
    print(f"查询结果文档列表：{result_str}")
    # query = "你想要查询的句子"
    # search_result = search_query(query, train_file_names, train_docs, stopwords, vectorizer, train_tfidf_matrix)
    # print("查询结果文档列表：")
    # for file_name in search_result:
    #     print(file_name)