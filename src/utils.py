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


def make_pattern(flag: str) -> Pattern:
    if not isinstance(flag, str):
        raise TypeError(f"Argument 'flag' must be a string, got {type(flag).__name__}.")

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
        raise TypeError(f"Argument 'text' must be a string, got {type(text).__name__}.")

    if not isinstance(section_patterns, dict):
        raise TypeError(
            f"Argument 'section_patterns' must be a dictionary, got {type(section_patterns).__name__}.")

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
                f"Argument 'section_names' must be a string or a list. Received type: {type(section_names).__name__}.")

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
        raise TypeError(f"Argument 'texts' must be a string list, got {type(texts).__name__}.")

    if not isinstance(mode, str):
        raise TypeError(f"Argument 'mode' must be a string, got {type(mode).__name__}.")

    if not isinstance(batch_size, int):
        raise TypeError(f"Argument 'batch_size' must be an integer, got {type(batch_size).__name__}.")

    if not isinstance(n_process, int):
        raise TypeError(f"Argument 'n_process' must be an integer, got {type(n_process).__name__}.")

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
            raise TypeError(f"Argument 'nlp_core' must be a spaCy Language model, got {type(nlp_core).__name__}.")

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
        raise TypeError(f"Argument 'df' must be a DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if label_col not in df.columns:
        raise ValueError(
            f"The specified label_col '{label_col}' is not in the DataFrame. Available columns: {list(df.columns)}")

    if group_col not in df.columns:
        raise ValueError(
            f"The specified group_col '{group_col}' is not in the DataFrame. Available columns: {list(df.columns)}")

    if not isinstance(n_splits, int):
        raise TypeError(f"Argument 'n_splits' must be a integer, got {type(n_splits).__name__}.")

    if n_splits < 3:
        raise ValueError(f"The value of 'n_splits' must be at least 3.")

    if not isinstance(seed, int):
        raise TypeError(f"Argument 'seed' must be a integer, got {type(seed).__name__}.")

    mlb = MultiLabelBinarizer()
    mlb.fit(df[label_col])

    num_labels = len(mlb.classes_)
    print(f"Total number of classes: {num_labels}")

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


def find_model_path(
        task: str,
        model: str,
        config: str,
        metric: str,
        with_drop: bool,
        only_t_s: bool,
        clean: bool):
    if not isinstance(task, str):
        raise TypeError(f"Argument 'task' must be a string, got {type(task).__name__}.")

    if not isinstance(model, str):
        raise TypeError(f"Argument 'model' must be a string, got {type(model).__name__}.")

    if not isinstance(config, str):
        raise TypeError(f"Argument 'config' must be a string, got {type(config).__name__}.")

    if not isinstance(metric, str):
        raise TypeError(f"Argument 'metric' must be a string, got {type(metric).__name__}.")

    if not isinstance(with_drop, bool):
        raise TypeError(f"Argument 'with_drop' must be a boolean, got {type(with_drop).__name__}.")

    if not isinstance(only_t_s, bool):
        raise TypeError(f"Argument 'only_t_s' must be a boolean, got {type(only_t_s).__name__}.")

    if not isinstance(clean, bool):
        raise TypeError(f"Argument 'bool' must be a boolean, got {type(clean).__name__}.")

    model_path = None
    model_dirs_path = Path(f"../models") / model / task
    metrics_files = list(model_dirs_path.resolve().rglob("evaluation_results/metrics.json"))

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
        model_path = model_dirs_path / ("with_dropped_sections" if with_drop else "without_dropped_sections")

        if only_t_s:
            if clean:
                if model == "finetuned":
                    model_path = model_path / "modernbert_cleaned_t_s_dataset"
                elif model == "baseline":
                    model_path = model_path / "tfidf_cleaned_t_s_dataset"
                else:
                    raise ValueError(f"Unknown model: {model}")
            else:
                model_path = model_path / "t_s_dataset"
        else:
            if clean:
                if model == "finetuned":
                    model_path = model_path / "modernbert_cleaned_base_dataset"
                elif model == "baseline":
                    model_path = model_path / "tfidf_cleaned_base_dataset"
                else:
                    raise ValueError(f"Unknown model: {model}")
            else:
                model_path = model_path / "base_dataset"

        if (model_path / "best_model").exists():
            model_path = model_path / "best_model"
        else:
            return None
    else:
        raise ValueError(f"Unknown config: {config}")

    return model_path


def preprocess_text(text: str,
                    model_path: Path,
                    nlp_core: Language,
                    t_s_config_data: dict):
    if not isinstance(text, str):
        raise TypeError(f"Argument 'text' must be a string, got {type(text).__name__}.")

    if not isinstance(model_path, Path):
        raise TypeError(f"Argument 'model_path' must be a Path, got {type(model_path).__name__}.")

    if not isinstance(nlp_core, Language):
        raise TypeError(f"Argument 'nlp_core' must be a spaCy Language model, got {type(nlp_core).__name__}.")

    if not isinstance(t_s_config_data, dict):
        raise TypeError(f"Argument 't_s_config_data' must be a dictionary, got {type(t_s_config_data).__name__}.")

    section_patterns = {
        key: [make_pattern(start_s), make_pattern(end_s)]
        for key, (start_s, end_s) in t_s_config_data["section_flags"].items()
    }

    path_str = str(model_path)

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
            text = clean_text([text], "tfidf", nlp_core)[0]

    return text



def predict_labels(
        text: str,
        task: str,
        model: str,
        config: str,
        free_text: bool,
        metric: str,
        nlp_core: Language,
        t_s_config_data: dict,
        with_drop: bool = None,
        only_t_s: bool = None,
        clean: bool = None,
        description_dir: Path = Path("../data/descriptions"),
):
    try:
        model_path = find_model_path(task, model, config, metric, with_drop, only_t_s, clean)
    except Exception as e:
        return {"error": f"Critical error during model search. Details: {str(e)}"}

    if model_path is None:
        return {"error": "Failed to find trained model with these settings."}

    if not free_text:
        try:
            text = preprocess_text(text, model_path, nlp_core, t_s_config_data)
        except Exception as e:
            return {"error": f"Critical error during text preprocessing. Details: {str(e)}"}

    try:
        mlb = joblib.load(model_path / "mlb.joblib")
    except Exception as e:
        return {"error": f"Failed to load the label encoder. Details: {str(e)}"}

    labels = {}

    if model == "baseline":
        try:
            pipeline = joblib.load(model_path / "pipeline.joblib")

            y_pred = pipeline.predict([text])

            labels["predicted_labels"] = mlb.inverse_transform(y_pred)[0]

            decision_scores = pipeline.decision_function([text])[0]

            raw_pairs = list(zip(mlb.classes_, decision_scores))

            sorted_raw = sorted(raw_pairs, key=lambda x: x[1], reverse=True)

            labels["labels_n_values"] = {label: float(val) for label, val in sorted_raw}
        except Exception as e:
            return {"error": f"Failed to run TF-IDF LinearSVC prediction. Details: {str(e)}"}
    elif model == "finetuned":
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.to(device)
            loaded_model.eval()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = loaded_model(**inputs)

            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            y_pred_binary = (probs >= 0.5).astype(int)

            labels["predicted_labels"] = mlb.inverse_transform(y_pred_binary.reshape(1, -1))[0]
            prob_pairs = list(zip(mlb.classes_, probs))

            sorted_probs = sorted(prob_pairs, key=lambda x: x[1], reverse=True)

            labels["labels_n_values"] = {label: float(val) for label, val in sorted_probs}

        except Exception as e:
            return {"error": f"Failed to run ModernBERT prediction. Details: {str(e)}"}

    return labels
