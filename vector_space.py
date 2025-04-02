import os
import math
import re
from collections import defaultdict, Counter


DATASET_PATH = os.path.join(os.path.dirname(__file__), 'ir')
# 단어들 추출해서 소문자로 변환하는 역할을 함.
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

# 파일들을 읽어서 단어들을 추출하는 역할을 함. 
def load_documents(path):
    documents = {}
    df = defaultdict(int)  # document frequency
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                words = tokenize(content)
                documents[filename] = words
                unique_terms = set(words)
                for term in unique_terms:
                    df[term] += 1
    return documents, df

# tf-idf를 통해 문서서 벡터 생성함.
def compute_tfidf(documents, df, N):
    tfidf_vectors = {}
    for doc_name, words in documents.items():
        tf = Counter(words)
        doc_vector = {}
        for term, freq in tf.items():
            idf = math.log(N / (df[term]))
            doc_vector[term] = (freq / len(words)) * idf
        tfidf_vectors[doc_name] = doc_vector
    return tfidf_vectors

# 벡터들 간에 cosine 유사도 계산하는 역할 함. 
def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[t] * vec2[t] for t in common_terms)

    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return numerator / (norm1 * norm2)

# 쿼리 벡터 생성하는 역할 함.
def build_query_vector(query, df, N):
    words = tokenize(query)
    tf = Counter(words)
    query_vector = {}
    for term, freq in tf.items():
        if term in df:
            idf = math.log(N / df[term])
            query_vector[term] = (freq / len(words)) * idf
    return query_vector

def main():
    print("Vector Space...")
    documents, df = load_documents(DATASET_PATH)
    N = len(documents)
    tfidf_vectors = compute_tfidf(documents, df, N)
    print("type the query. (ex: apple banana)")

    while True:
        try:
            query = input("\n>>> ").strip()
            if query.lower() in ('exit', 'quit'):
                print("exited.")
                break

            query_vector = build_query_vector(query, df, N)
            scores = []
            for doc_name, doc_vector in tfidf_vectors.items():
                sim = cosine_similarity(query_vector, doc_vector)
                scores.append((doc_name, sim))

            top_docs = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
            if any(score > 0 for _, score in top_docs):
                print("relevant txt file Top 10:")
                for rank, (doc, score) in enumerate(top_docs, 1):
                    if score > 0:
                        print(f"{rank}. {doc} (similarity: {score:.4f})")
            else:
                print("no result")

        except KeyboardInterrupt:
            print("\nexited.")
            break

if __name__ == '__main__':
    main()
