import os
import json
from boolean_model import build_inverted_index, parse_query, infix_to_postfix, evaluate_postfix, DATASET_PATH

GT_PATH = os.path.join(os.path.dirname(__file__), 'gt')

def load_ground_truth(query):
    filename = query.strip().replace(" ", "_") + ".txt"
    gt_file = os.path.join(GT_PATH, filename)
    if not os.path.exists(gt_file):
        return None
    with open(gt_file, 'r', encoding='utf-8') as f:
        return set(json.load(f))

def evaluate(retrieved, ground_truth):
    retrieved = set(retrieved)
    relevant = set(ground_truth)

    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1, tp,fp,fn

def main():
    print("Evaluation mode for Boolean Model")
    print("type 'exit' to quit")

    index, all_docs = build_inverted_index(DATASET_PATH)

    while True:
        query = input("\nEnter query ex) ( apple AND orange ) OR grape : ").strip()
        if query.lower() in ('exit', 'quit'):
            print("Bye!")
            break

        try:
            tokens = parse_query(query)
            postfix = infix_to_postfix(tokens)
            result_docs = evaluate_postfix(postfix, index, all_docs)

            print(f"\nRetrieved {len(result_docs)} document(s):")
            for doc in sorted(result_docs):
                print(f" - {doc}")

            gt = load_ground_truth(query)
            if gt is None:
                print("\n⚠️ Ground truth not found for this query.")
            else:
                # ✅ 정답과 일치한 문서 보여주기
                matched = sorted(set(result_docs) & gt)
                if matched:
                    print(f"\n✅ Matched {len(matched)} document(s):")
                    for doc in matched:
                        print(f" ✔ {doc}")
                else:
                    print("\n❌ No matched documents with ground truth.")
                precision, recall, f1, tp, fp, fn = evaluate(result_docs, gt)
                print("\n📊 Evaluation:")
                print(f" - TP (True Positives):  {tp}")
                print(f" - FP (False Positives): {fp}")
                print(f" - FN (False Negatives): {fn}")
                print(f" - Precision: {precision:.4f}")
                print(f" - Recall:    {recall:.4f}")
                print(f" - F1 Score:  {f1:.4f}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
