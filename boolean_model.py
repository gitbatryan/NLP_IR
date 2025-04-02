import os
import re
from collections import defaultdict

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')

#단어들 추출해서 소문자로 변환하는 역할함.
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

#dataset 폴더안에 있는 txt문서들 읽고, 단어가 등장하는 문서를 알려줌. 
def build_inverted_index(path):
    inverted_index = defaultdict(set)
    all_docs = set()

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            all_docs.add(filename)
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                words = tokenize(content)
                for word in set(words):
                    inverted_index[word].add(filename)

    return inverted_index, all_docs

# 컴퓨터가 이해하기 쉽게 infix를 postfix로 변환
def infix_to_postfix(query_tokens):
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    stack = []

    for token in query_tokens:
        token = token.upper()
        if token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # '(' 제거
        elif token in precedence:
            while (stack and stack[-1] != '(' and
                   precedence.get(stack[-1], 0) >= precedence[token]):
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token.lower())  # 검색어는 소문자
    while stack:
        output.append(stack.pop())

    return output

# Postfix를 읽고 검색을 수행함.
def evaluate_postfix(postfix_tokens, index, all_docs):
    stack = []
    for token in postfix_tokens:
        if token == 'AND':
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif token == 'OR':
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        elif token == 'NOT':
            a = stack.pop()
            stack.append(all_docs - a)
        else:  # 일반 단어
            stack.append(index.get(token, set()))
    return stack[0] if stack else set()

def parse_query(query):
    # 사용자의 입력을 단어, 괄호 이렇게 분리함. 
    return re.findall(r'\b\w+\b|[()]', query)

def main(): #메인
    print("Boolean model (AND, OR, NOT, parentheses can be used)")
    index, all_docs = build_inverted_index(DATASET_PATH)
    print("type 'exit' to quit")

    while True:
        try:
            query = input("\n>>> ").strip()
            if query.lower() in ('exit', 'quit'):
                print("exited")
                break
            tokens = parse_query(query)
            postfix = infix_to_postfix(tokens)
            result = evaluate_postfix(postfix, index, all_docs)
            if result:
                print("result :")
                for doc in sorted(result):
                    print(f" - {doc}")
            else:
                print("no result.")
        except Exception as e:
            print(f"error: {e}")

if __name__ == '__main__':
    main()
