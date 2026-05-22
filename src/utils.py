from itertools import combinations
import re
from typing import Pattern
from tqdm.notebook import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from spacy import Language
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    classification_report, hamming_loss
from transformers import Trainer
from scipy.special import expit
from transformers_interpret import MultiLabelClassificationExplainer


def make_pattern(flag: str) -> Pattern:
    if not isinstance(flag, str):
        raise TypeError(f"Parameter 'flag' must be a string, got {type(flag).__name__}.")

    words = flag.split()
    n = len(words)

    max_masked_count = n - 1 if n > 1 else 0

    variations = []

    for i in range(max_masked_count + 1):
        for words_to_mask in combinations(range(n), i):
            current_words = []
            for j, word in enumerate(words):
                if j in words_to_mask:
                    current_words.append("___")
                else:
                    current_words.append(re.escape(word))

            base_string = r"\s+".join(current_words)
            pattern_string = base_string + r"\:"
            variations.append(pattern_string)

    final_pattern = f"(?:{'|'.join(variations)})"
    return re.compile(final_pattern)


def section_manager(text: str,
                    section_patterns: dict,
                    task: str = "check",
                    section_names: str | list | None = None) -> str | bool:
    if not isinstance(text, str):
        raise TypeError(f"Parameter 'text' must be a string, got {type(text).__name__}.")

    if not isinstance(section_patterns, dict):
        raise TypeError(
            f"Parameter 'section_patterns' must be a dictionary, got {type(section_patterns).__name__}.")

    if task == "check":

        i = 0

        for key, (start_flag, end_flag) in section_patterns.items():

            start_match = start_flag.search(text, pos=i)

            if not start_match:
                return False
            end_match = end_flag.search(text, pos=start_match.end())

            if not end_match:
                return False

            i = end_match.start()

        return True

    elif task in ["drop", "keep"]:

        if isinstance(section_names, str):
            target_sections = [section_names]
        elif isinstance(section_names, list):
            target_sections = section_names
        elif section_names is None:
            raise ValueError(f"No section name provided for '{task}' task")
        else:
            raise TypeError(
                f"Parameter 'section_names' must be a string or a list. Received type: {type(section_names).__name__}.")

        if len(target_sections) != len(set(target_sections)):
            raise ValueError("Duplicate entries found in 'section_names'.")

        if len(section_patterns.keys()) < len(target_sections):
            raise ValueError(
                f"The number of names in 'section_names' ({len(target_sections)}) exceeds the total number of keys in "
                f"'section_patterns' ({len(section_patterns)}).")

        keys = section_patterns.keys()
        for name in target_sections:
            if name not in keys:
                raise ValueError(
                    f"Section name '{name}' not found in section patterns keys. Available keys: {list(keys)}.")

        arr = []
        i = 0

        for key, (start_flag, end_flag) in section_patterns.items():

            start_match = start_flag.search(text, pos=i)

            if not start_match:
                raise ValueError(f"Could not find starting flag of section '{key}'.")

            end_match = end_flag.search(text, pos=start_match.end())

            if not end_match:
                raise ValueError(f"Could not find ending flag of section '{key}'.")

            if key in target_sections:
                arr.append({
                    "key": key,
                    "full_start": start_match.start(),
                    "content_start": start_match.end(),
                    "end": end_match.start()
                })

            i = end_match.start()

        if task == "drop":

            arr.sort(key=lambda x: x["full_start"], reverse=True)
            cleaned_text = text

            for item in arr:
                cleaned_text = cleaned_text[:item["full_start"]] + cleaned_text[item["end"]:]

            return cleaned_text

        elif task == "keep":

            arr.sort(key=lambda x: target_sections.index(x["key"]))

            kept_parts = []
            for item in arr:
                content = text[item["content_start"]:item["end"]].strip()
                kept_parts.append(f"{item['key']}:\n{content}")

            return "\n \n".join(kept_parts)
    else:
        raise ValueError(f"Unknown task: {task}")


def clean_text(texts: list[str], mode: str = "modernbert", nlp_core: Language = None,
               batch_size: int = 32, n_process: int = 4) -> list[str]:
    if not isinstance(texts, list):
        raise TypeError(f"Parameter 'texts' must be a string list, got {type(texts).__name__}.")

    if not isinstance(mode, str):
        raise TypeError(f"Parameter 'mode' must be a string, got {type(mode).__name__}.")

    if not isinstance(batch_size, int):
        raise TypeError(f"Parameter 'batch_size' must be an integer, got {type(batch_size).__name__}.")

    if not isinstance(n_process, int):
        raise TypeError(f"Parameter 'n_process' must be an integer, got {type(n_process).__name__}.")

    cleaned_texts = []

    long_special_chars = re.compile(r"[^a-zA-Z0-9\s]{4,}")
    non_alphanumeric = re.compile(r"[^a-z0-9]")
    lonely_numeric = re.compile(r"\b\d+\b")
    multiple_whitespaces = re.compile(r"\s+")

    if mode == "modernbert":

        for text in tqdm(texts):
            t = long_special_chars.sub("", text)
            t = multiple_whitespaces.sub(" ", t).strip()

            cleaned_texts.append(t)

    elif mode == "tfidf":

        if not isinstance(nlp_core, Language):
            raise TypeError(f"Parameter 'nlp_core' must be a spaCy Language model, got {type(nlp_core).__name__}.")

        for doc in tqdm(nlp_core.pipe(texts, batch_size=batch_size, n_process=n_process, disable=["parser", "ner"]),
                        total=len(texts)):
            tokens = []

            for token in doc:
                if token.is_stop or token.is_punct or token.is_space:
                    continue

                t = token.lemma_.lower().strip()

                t = non_alphanumeric.sub(" ", t)

                t = lonely_numeric.sub("", t)

                t = multiple_whitespaces.sub(" ", t).strip()

                if t:
                    tokens.append(t)

            cleaned_texts.append(" ".join(tokens))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cleaned_texts


def split_dataframe(df: pd.DataFrame,
                    label_col: str,
                    group_col: str,
                    n_splits: int = 5,
                    seed: int = 42):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Parameter 'df' must be a DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if label_col not in df.columns:
        raise ValueError(
            f"The specified label_col '{label_col}' is not in the DataFrame. Available columns: {list(df.columns)}")

    if group_col not in df.columns:
        raise ValueError(
            f"The specified group_col '{group_col}' is not in the DataFrame. Available columns: {list(df.columns)}")

    if not isinstance(n_splits, int):
        raise TypeError(f"Parameter 'n_splits' must be a integer, got {type(n_splits).__name__}.")

    if n_splits < 3:
        raise ValueError(f"The value of 'n_splits' must be at least 3.")

    if not isinstance(seed, int):
        raise TypeError(f"Parameter 'seed' must be a integer, got {type(seed).__name__}.")

    mlb = MultiLabelBinarizer()
    mlb.fit(df[label_col])

    num_labels = len(mlb.classes_)
    print(f"Total number of labels: {num_labels}")

    group_df = df.groupby(group_col)[label_col].apply(
        lambda x: list(set([item for sublist in x for item in sublist]))).to_frame()

    group_ids = group_df.index.values
    patients_y = mlb.transform(group_df[label_col])

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = list(mskf.split(group_ids, patients_y))

    test_fold_indices = splits[0][1]
    val_fold_indices = splits[1][1]

    test_group_ids = group_ids[test_fold_indices]
    val_group_ids = group_ids[val_fold_indices]

    all_subject_ids_set = set(group_ids)
    test_ids_set = set(test_group_ids)
    val_ids_set = set(val_group_ids)
    train_ids_set = all_subject_ids_set - test_ids_set - val_ids_set

    df_train = df[df[group_col].isin(train_ids_set)].reset_index(drop=True)
    df_val = df[df[group_col].isin(val_ids_set)].reset_index(drop=True)
    df_test = df[df[group_col].isin(test_ids_set)].reset_index(drop=True)

    datasets = [("Train", df_train), ("Val", df_val), ("Test", df_test)]

    for name, dataset in datasets:
        y_matrix = mlb.transform(dataset[label_col])

        labels_per_row = y_matrix.sum(axis=1)
        present_classes = np.count_nonzero(y_matrix.sum(axis=0))
        coverage = (present_classes / num_labels) * 100

        print(f"\n{name} dataset summary:")
        print(f"Number of rows: {len(dataset)}")
        print(f"Unique patients: {dataset[group_col].nunique()}")
        print(f"Average labels per row: {labels_per_row.mean():.2f}")
        print(f"Label coverage: {present_classes}/{num_labels} ({coverage:.1f}%)")

        counts = y_matrix.sum(axis=0)
        percentages = (counts / len(dataset)) * 100

        stats_df = pd.DataFrame({
            "label": mlb.classes_,
            "count": counts,
            "percentage": percentages
        })

        stats_df = stats_df.sort_values(by="count", ascending=False)

        stats_df["percentage"] = stats_df["percentage"].map('{:.2f}%'.format)

        print(stats_df.to_string(index=False))

    return df_train, df_val, df_test, mlb, num_labels


def evaluate_finetuned(model, tokenizer, training_args, dataset, threshold, mlb):
    eval_trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=dataset,
    )

    results = eval_trainer.predict(dataset)

    logits = results.predictions
    labels = results.label_ids

    probs = expit(logits)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(labels, preds),
        "test_micro_f1": f1_score(labels, preds, average='micro', zero_division=0),
        "test_macro_f1": f1_score(labels, preds, average='macro', zero_division=0),
        "test_weighted_f1": f1_score(labels, preds, average='weighted', zero_division=0),
        "test_samples_f1": f1_score(labels, preds, average='samples', zero_division=0),
        "test_micro_precision": precision_score(labels, preds, average='micro', zero_division=0),
        "test_macro_precision": precision_score(labels, preds, average='macro', zero_division=0),
        "test_micro_recall": recall_score(labels, preds, average='micro', zero_division=0),
        "test_macro_recall": recall_score(labels, preds, average='macro', zero_division=0),
        "test_micro_auc": roc_auc_score(labels, probs, average='micro'),
        "test_macro_auc": roc_auc_score(labels, probs, average='macro'),
        "test_hamming_loss": hamming_loss(y_true=labels, y_pred=preds)
    }

    report = classification_report(
        labels,
        preds,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True
    )

    report = pd.DataFrame(report).transpose()
    summary_metrics = ["micro avg", "macro avg", "weighted avg", "samples avg"]
    main_report = report.drop(index=[i for i in summary_metrics if i in report.index])
    summary_report = report.loc[[i for i in summary_metrics if i in report.index]]
    main_report = main_report.sort_values(by="f1-score", ascending=False)

    report = pd.concat([main_report, summary_report])
    metrics = pd.DataFrame([metrics])

    return metrics, report, labels, probs


def evaluate_baseline(model, X, y, mlb, threshold=0.5):
    y_scores = model.predict_proba(X)

    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_micro_f1": f1_score(y, y_pred, average='micro', zero_division=0),
        "test_macro_f1": f1_score(y, y_pred, average='macro', zero_division=0),
        "test_samples_f1": f1_score(y, y_pred, average='samples', zero_division=0),
        "test_weighted_f1": f1_score(y, y_pred, average='weighted', zero_division=0),
        "test_micro_precision": precision_score(y, y_pred, average='micro', zero_division=0),
        "test_macro_precision": precision_score(y, y_pred, average='macro', zero_division=0),
        "test_micro_recall": recall_score(y, y_pred, average='micro', zero_division=0),
        "test_macro_recall": recall_score(y, y_pred, average='macro', zero_division=0),
        "test_micro_auc": roc_auc_score(y, y_scores, average='micro'),
        "test_macro_auc": roc_auc_score(y, y_scores, average='macro'),
        "test_hamming_loss": hamming_loss(y_true=y, y_pred=y_pred)
    }

    report = classification_report(
        y,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    summary_metrics = ["micro avg", "macro avg", "weighted avg", "samples avg"]
    main_report = report_df.drop(index=[i for i in summary_metrics if i in report_df.index])
    summary_report = report_df.loc[[i for i in summary_metrics if i in report_df.index]]
    main_report = main_report.sort_values(by="f1-score", ascending=False)

    report = pd.concat([main_report, summary_report])
    metrics = pd.DataFrame([metrics])

    return metrics, report, y, y_scores


def find_optimal_thresholds(y_true, y_probs, mlb):
    n_classes = y_true.shape[1]

    best_thresholds = np.full(n_classes, 0.5)
    threshold_candidates = np.linspace(0.00, 1.00, 21)

    for i in tqdm(range(n_classes)):
        y_true_col = y_true[:, i]
        y_prob_col = y_probs[:, i]

        class_name = mlb.classes_[i]

        y_pred_05 = (y_prob_col >= 0.5).astype(int)
        score_at_05 = f1_score(y_true_col, y_pred_05)

        best_score = -1
        best_t = 0.5

        for t in threshold_candidates:
            y_pred_t = (y_prob_col >= t).astype(int)
            score = f1_score(y_true_col, y_pred_t)

            if score > best_score:
                best_score = score
                best_t = t

        best_thresholds[i] = best_t

        tqdm.write(
            f"{class_name:<10} | Opt. threshold: {best_t:.3f} | "
            f"F1 Score(0.5): {score_at_05:.5f} -> Best F1 Score: {best_score:.5f}"
        )

    return best_thresholds


def find_model_path(
        task: str,
        model: str,
        config: str,
        metric: str,
        with_drop: bool,
        only_t_s: bool,
        clean: bool,
        weighted: bool,
        thrs_tuned: bool
):
    if not isinstance(task, str):
        raise TypeError(f"Parameter 'task' must be a string, got {type(task).__name__}.")

    if not isinstance(model, str):
        raise TypeError(f"Parameter 'model' must be a string, got {type(model).__name__}.")

    if not isinstance(config, str):
        raise TypeError(f"Parameter 'config' must be a string, got {type(config).__name__}.")

    if not isinstance(metric, str):
        raise TypeError(f"Parameter 'metric' must be a string, got {type(metric).__name__}.")


    model_path = None
    model_dirs_path = Path(f"../models") / model / task
    metrics_files = list(model_dirs_path.resolve().rglob("val_results/metrics.json"))

    if config == "best":
        best_f1 = 0

        for metrics_file in metrics_files:
            df_metrics = pd.read_json(metrics_file)

            if metric in df_metrics.columns:
                current_f1 = df_metrics[metric].iloc[0]

                potential_model = metrics_file.parent.parent / "best_model"

                if current_f1 > best_f1 and potential_model.exists():
                    best_f1 = current_f1
                    model_path = potential_model

    elif config == "worst":
        worst_f1 = 1.0

        for metrics_file in metrics_files:
            df_metrics = pd.read_json(metrics_file)

            if metric in df_metrics.columns:
                current_f1 = df_metrics[metric].iloc[0]

                potential_model = metrics_file.parent.parent / "best_model"

                if current_f1 < worst_f1 and potential_model.exists():
                    worst_f1 = current_f1
                    model_path = potential_model


    elif config == "custom":

        sub_dir = "with_dropped_sections" if with_drop else "without_dropped_sections"

        base_dir = model_dirs_path / sub_dir

        dataset_type = "t_s_dataset" if only_t_s else "base_dataset"

        if clean:

            prefix = "modernbert_cleaned" if model == "finetuned" else "tfidf_cleaned"

            folder_name = f"{prefix}_{dataset_type}"

        else:

            folder_name = dataset_type

        if weighted:
            folder_name += "_weighted"

        if thrs_tuned:
            folder_name += "_threshold_tuned"

        model_path = base_dir / folder_name

        if (model_path / "best_model").exists():

            model_path = model_path / "best_model"

        else:

            return None
    else:
        raise ValueError(f"Unknown config: {config}")

    return model_path


def preprocess_text(text: str,
                    free_text: bool,
                    model_path: Path,
                    nlp_core: Language,
                    t_s_config_data: dict):
    if not isinstance(text, str):
        raise TypeError(f"Parameter 'text' must be a string, got {type(text).__name__}.")

    if not isinstance(model_path, Path):
        raise TypeError(f"Parameter 'model_path' must be a Path, got {type(model_path).__name__}.")

    if not isinstance(nlp_core, Language):
        raise TypeError(f"Parameter 'nlp_core' must be a spaCy Language model, got {type(nlp_core).__name__}.")

    if not isinstance(t_s_config_data, dict):
        raise TypeError(f"Parameter 't_s_config_data' must be a dictionary, got {type(t_s_config_data).__name__}.")

    section_patterns = {
        key: [make_pattern(start_s), make_pattern(end_s)]
        for key, (start_s, end_s) in t_s_config_data["section_flags"].items()
    }

    path_str = str(model_path)

    if not free_text:
        if "with_dropped_sections" in path_str:
            if "t_s" in path_str:
                text = section_manager(text, section_patterns, "keep",
                                       t_s_config_data["with_dropped_sections"]["sections_to_keep"])
            else:
                text = section_manager(text, section_patterns, "drop",
                                       t_s_config_data["with_dropped_sections"]["sections_to_drop"])
        else:
            if "t_s" in path_str:
                text = section_manager(text, section_patterns, "keep",
                                       t_s_config_data["without_dropped_sections"]["sections_to_keep"])
    if "cleaned" in path_str:
        if "modernbert" in path_str:
            text = clean_text([text], "modernbert")[0]
        elif "tfidf" in path_str:
            text = clean_text([text], "tfidf", nlp_core, n_process=1)[0]

    return text


def generate_heatmaps(xai_dict):
    html_outputs = {}

    for label, token_scores in xai_dict.items():
        valid_scores = []

        for word, score in token_scores:

            if word:
                valid_scores.append(abs(score))

        max_abs_score = max(valid_scores) if valid_scores else 1.0
        if max_abs_score == 0.0:
            max_abs_score = 1.0

        html_content = ('<div style="font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px; '
                        'background-color: #ffffff; color: #000000; padding: 15px; border: 1px solid #eee; '
                        'border-radius: 5px;">')

        for word, score in token_scores:

            if word in {"[CLS]", "[SEP]"}:
                continue

            relative_score = abs(score) / max_abs_score

            alpha = min(relative_score, 0.7)

            if score > 0:
                bg_color = f"rgba(0, 255, 0, {alpha:.3f})"
            elif score < 0:
                bg_color = f"rgba(255, 0, 0, {alpha:.3f})"
            else:
                bg_color = "transparent"

            text_style = "color: #000;"

            html_content += (f'<span style="background-color: {bg_color}; {text_style} padding: 1px 0px; '
                             f'border-radius: 2px;">{word}</span> ')

        html_content += '</div>'

        html_outputs[label] = html_content

    return html_outputs


def get_shap_values(text, pipeline, expected_values, mlb_classes):

    vectorizer = pipeline.named_steps['tfidf']
    selector = pipeline.named_steps['selector']
    classifier = pipeline.named_steps['clf']

    X_tfidf = vectorizer.transform([text])
    X_tfidf = selector.transform(X_tfidf)
    x_instance = X_tfidf.toarray()[0]

    feature_names = vectorizer.get_feature_names_out()
    support = selector.get_support()
    selected_feature_names = feature_names[support]

    vocab = {name: i for i, name in enumerate(selected_feature_names)}

    tokens = re.findall(r'\b\w+\b|[^\w\s]|\s+', text)

    xai_dict = {}

    for i, label in enumerate(mlb_classes):
        estimator = classifier.estimators_[i]

        if hasattr(estimator.coef_, 'toarray'):
            beta = estimator.coef_.toarray()[0]
        else:
            beta = estimator.coef_[0]

        shap_values = beta * (x_instance - expected_values)

        token_scores = []
        for token in tokens:
            token_chk = token.strip().lower()

            if token_chk and token_chk in vocab:
                idx = vocab[token_chk]
                score = float(shap_values[idx])
            else:
                score = 0.0

            token_scores.append((token, score))

        xai_dict[label] = token_scores

    return xai_dict


def predict_labels(
        text: str,
        task: str,
        model_type: str,
        config: str,
        free_text: bool,
        metric: str,
        nlp_core: Language,
        t_s_config_data: dict,
        with_drop: bool = None,
        only_t_s: bool = None,
        clean: bool = None,
        weighted: bool = None,
        thrs_tuned: bool = None
):
    try:
        model_path = find_model_path(task, model_type, config, metric, with_drop, only_t_s, clean, weighted, thrs_tuned)
    except Exception as e:
        return {"error": f"Critical error during model search. Details: {str(e)}"}

    if model_path is None:
        return {"error": "Failed to find trained model with these settings."}

    try:
        text = preprocess_text(text, free_text, model_path, nlp_core, t_s_config_data)
    except Exception as e:
        return {"error": f"Critical error during text preprocessing. Details: {str(e)}"}

    try:
        mlb = joblib.load(model_path / "mlb.joblib")
    except Exception as e:
        return {"error": f"Failed to load the label encoder. Details: {str(e)}"}

    labels = {}

    if model_type == "baseline":
        try:
            pipeline = joblib.load(model_path / "pipeline.joblib")

            expected_values = joblib.load(model_path / "expected_values.joblib")

            threshold_file = model_path / "thresholds.json"

            if threshold_file.exists():
                with open(threshold_file, "r") as f:
                    loaded_thresholds = np.array(json.load(f))
            else:
                loaded_thresholds = np.full(len(mlb.classes_), 0.5)

            probs = pipeline.predict_proba([text])[0]

            y_pred = (probs >= loaded_thresholds).astype(int)

            labels["predicted_labels"] = mlb.inverse_transform(y_pred.reshape(1, -1))[0]

            prob_triplets = list(zip(mlb.classes_, probs, loaded_thresholds))

            sorted_probs = sorted(prob_triplets, key=lambda x: x[1], reverse=True)

            labels["labels_n_values"] = {
                label: (float(val), float(thr))
                for label, val, thr in sorted_probs
            }

            labels["heatmaps"] = generate_heatmaps(get_shap_values(text, pipeline, expected_values, mlb.classes_))


        except Exception as e:
            return {"error": f"Failed to run TF-IDF Logistic Regression prediction. Details: {str(e)}"}
    elif model_type == "finetuned":
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            id2label = {i: label for i, label in enumerate(mlb.classes_)}
            label2id = {label: i for i, label in enumerate(mlb.classes_)}

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                attn_implementation="eager",
                id2label=id2label,
                label2id=label2id
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            with open(model_path / "inference_config.json", "r") as f:
                loaded_config = json.load(f)

            tokenized_input = tokenizer(text,
                                        return_tensors="pt",
                                        truncation=loaded_config["TRUNCATION"],
                                        max_length=loaded_config["MAX_LENGTH"])

            cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)
            truncated_text = tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

            labels["heatmaps"] = generate_heatmaps(cls_explainer(truncated_text))

            threshold_file = model_path / "thresholds.json"

            if threshold_file.exists():
                with open(threshold_file, "r") as f:
                    loaded_thresholds = np.array(json.load(f))
            else:
                loaded_thresholds = np.full(len(mlb.classes_), 0.5)

            model.eval()

            model_input = {k: v.to(device) for k, v in tokenized_input.items()}

            with torch.no_grad():
                outputs = model(**model_input)

            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            y_pred_binary = (probs >= loaded_thresholds).astype(int)

            labels["predicted_labels"] = mlb.inverse_transform(y_pred_binary.reshape(1, -1))[0]

            prob_triplets = list(zip(mlb.classes_, probs, loaded_thresholds))

            sorted_probs = sorted(prob_triplets, key=lambda x: x[1], reverse=True)

            labels["labels_n_values"] = {
                label: (float(val), float(thr))
                for label, val, thr in sorted_probs
            }

        except Exception as e:
            return {"error": f"Failed to run ModernBERT prediction. Details: {str(e)}"}

    return labels


def get_description(label, chapter_df, code_df):
    match = chapter_df[chapter_df["chapter"] == label]
    if not match.empty:
        return match.iloc[0]["long_title"]

    match = code_df[code_df["icd_code"] == label]
    if not match.empty:
        return match.iloc[0]["long_title"]

    return f"Description not found."
