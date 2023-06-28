from collections import Counter
import string
import re
import json

from rouge import Rouge

rouge = Rouge()


def rouge_score(pred, gold):
    # bypass the ValueError "Hypothesis is empty," I love this exception!
    try:
        scores = rouge.get_scores(pred, gold)[0]
    except ValueError:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}, }
    return scores


def rouge_scores(preds, golds):
    return [rouge_score(p, g) for p, g in zip(preds, golds)]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_scores(preds, golds):
    return [f1_score(p, g) for p, g in zip(preds, golds)]


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_scores(preds, golds):
    return [exact_match_score(p, g) for p, g in zip(preds, golds)]


def _get_scores(answers, refs, fn):
    return [metric_max_over_ground_truths(fn, pred, rs) for pred, rs in zip(answers, refs)]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_references(fi, sep='\t'):
    def parse_pandas_answer(a_string):
        # to avoid a pandas dependency, deserialize these manually
        try:
            parsed_answers = eval(a_string) if a_string.startswith('[') else eval(a_string.replace('""', '"')[1:-1])
        except:
            parsed_answers = eval(a_string.replace('""', '"').replace('""', '"').replace('""', '"')[1:-1])
        return parsed_answers

    references = []
    for i, line in enumerate(open(fi)):
        q, answer_str = line.strip('\n').split(sep)
        refs = parse_pandas_answer(answer_str)
        references.append({'references': refs, 'id': i})
    return references


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def read_json(path):
    with open(path) as f:
        data = json.load(f)
        assert all(key.endswith(str(i)) for i, key in enumerate(data))
        return [{'id': i, 'prediction': data[key]} for i, key in enumerate(data)]


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f]


def read_annotations(annotations_data_path):
    return read_jsonl(annotations_data_path)


def read_predictions(path):
    if path.endswith('.jsonl'):
        return read_jsonl(path)
    elif path.endswith('json'):
        return read_json(path)
    else:
        return [{'id': i, 'prediction': pred} for i, pred in enumerate(read_lines(path))]


def _print_score(label, results_dict):
    print('-' * 50)
    print('Label       :', label)
    # print('N examples  : ', results_dict['n_examples'])
    print('Exact Match : ', results_dict['exact_match'])
    print('F1 score    : ', results_dict['f1_score'])


def get_out_of_context_prediction(dev_file_path, dev_prediction_path):
    with open(dev_file_path) as f:
        dev_data = json.load(f)
        context_dic = {data["id"]: data["context"] for data in dev_data}
    with open(dev_prediction_path) as f:
        dev_predictions = json.load(f)
    bad_ids = []
    for id_ in dev_predictions:
        pred_answer = dev_predictions[id_]
        if pred_answer not in context_dic[id_]:
            print(id_, pred_answer)
            bad_ids.append(id_)
    print(bad_ids)
