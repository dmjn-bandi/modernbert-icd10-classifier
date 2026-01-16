import gradio as gr
import json
import pandas as pd
from src.utils import predict_labels
import en_core_sci_sm

NLP_CORE = en_core_sci_sm.load()

with open("../t_s_config.json", "r", encoding="utf-8") as f:
    T_S_CONFIG_DATA = json.load(f)

CHAPTER_DESCRIPTIONS = pd.read_json("../data/descriptions/icd_chapter_descriptions.json")
CODE_DESCRIPTIONS = pd.read_json("../data/descriptions/icd_code_descriptions.json")

with gr.Blocks() as demo:
    with gr.Group():
        with gr.Row():
            with gr.Column():
                tb_text = gr.Textbox(
                    lines=15,
                    placeholder="",
                    label="Discharge Summary",
                    interactive=True
                )
                chk_free_text = gr.Checkbox(label="Free Text Input", interactive=True)

        with gr.Row():
            with gr.Column():
                dd_task = gr.Dropdown(
                    choices=[
                        ("ICD-10-CM Frequent Chapter Classification", "frequent_chapter"),
                        ("ICD-10-CM Top 50 Code Classification", "top_50_code")],
                    value="frequent_chapter",
                    label="Task",
                    interactive=True,
                    filterable=False)
                dd_model = gr.Dropdown(
                    choices=[
                        ("TF-IDF LinearSVC", "baseline"),
                        ("ModernBERT", "finetuned")],
                    value="finetuned",
                    label="Model",
                    interactive=True,
                    filterable=False)
                dd_config = gr.Dropdown(
                    choices=[
                        ("Best", "best"),
                        ("Worst", "worst"),
                        ("Custom", "custom")],
                    value="best",
                    label="Config",
                    interactive=True,
                    filterable=False)
                with gr.Column(visible=True) as group_metric:
                    dd_metric = gr.Dropdown(choices=[
                        ("Accuracy", "test_accuracy"),
                        ("Micro F1", "test_micro_f1"),
                        ("Macro F1", "test_macro_f1"),
                        ("Weighted F1", "test_weighted_f1"),
                        ("Samples F1", "test_samples_f1"),
                        ("Micro Precision", "test_micro_precision"),
                        ("Macro Precision", "test_macro_precision"),
                        ("Micro Recall", "test_micro_recall"),
                        ("Macro Recall", "test_macro_recall"),
                        ("Micro Auc", "test_micro_auc"),
                        ("Macro Auc", "test_macro_auc")],
                        value="test_micro_f1",
                        label="Metric",
                        interactive=True,
                        filterable=False)
                with gr.Column(visible=False) as group_custom:
                    chk_with_drop = gr.Checkbox(label="With Dropped Sections", interactive=True)
                    chk_only_t_s = gr.Checkbox(label="Only Target Sections", interactive=True)
                    chk_clean = gr.Checkbox(label="Cleaned Text", interactive=True)

    btn_predict = gr.Button("Predict Labels", variant="primary")

    df_output = gr.Dataframe(
        headers=["Label", "Description"],
        datatype="str",
        label="Predicted Labels",
        interactive=False,
        wrap=True
    )

    plot_output = gr.BarPlot(
        x="score",
        y="label",
        color="predicted",
        color_map={"True": "green", "False": "red"},
        tooltip=["label", "score", "predicted", "description"],
    )


    def toggle_config(config_value):

        if config_value == "custom":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)


    dd_config.change(
        fn=toggle_config,
        inputs=dd_config,
        outputs=[group_metric, group_custom]
    )


    def predict_labels_wrapper(
            tb_text: str,
            dd_task: str,
            dd_model: str,
            dd_config: str,
            chk_free_text: bool,
            dd_metric: str,
            chk_with_drop: bool,
            chk_only_t_s: bool,
            chk_clean: bool):

        if not tb_text.strip():
            raise gr.Error("No input text provided.")

        result = predict_labels(
            text=tb_text,
            task=dd_task,
            model=dd_model,
            config=dd_config,
            free_text=chk_free_text,
            metric=dd_metric,
            nlp_core=NLP_CORE,
            t_s_config_data=T_S_CONFIG_DATA,
            with_drop=chk_with_drop,
            only_t_s=chk_only_t_s,
            clean=chk_clean,
        )

        if "error" in result:
            raise gr.Error(result["error"])

        labels_list = result.get("predicted_labels", [])
        labels_values = result.get("labels_n_values", {})

        table_data = []
        for label in labels_list:
            match = CHAPTER_DESCRIPTIONS[CHAPTER_DESCRIPTIONS["chapter"] == label]
            if not match.empty:
                description = match.iloc[0]["long_title"]
            else:
                match = CODE_DESCRIPTIONS[CODE_DESCRIPTIONS["icd_code"] == label]
                if not match.empty:
                    description = match.iloc[0]["long_title"]
                else:
                    description = f"Description not found for: {label}"
            table_data.append([label, description])

        df_table = pd.DataFrame(table_data, columns=["Label", "Description"])

        data = []
        for label, score in labels_values.items():
            predicted = "True" if label in labels_list else "False"

            match = CHAPTER_DESCRIPTIONS[CHAPTER_DESCRIPTIONS["chapter"] == label]
            if not match.empty:
                description = match.iloc[0]["long_title"]
            else:
                match = CODE_DESCRIPTIONS[CODE_DESCRIPTIONS["icd_code"] == label]
                if not match.empty:
                    description = match.iloc[0]["long_title"]
                else:
                    description = f"Description not found for: {label}"

            data.append({
                "label": label,
                "score": round(score, 3),
                "predicted": predicted,
                "description": description
            })

        df_plot = pd.DataFrame(data)

        return df_plot, df_table


    btn_predict.click(
        fn=predict_labels_wrapper,
        inputs=[
            tb_text,
            dd_task,
            dd_model,
            dd_config,
            chk_free_text,
            dd_metric,
            chk_with_drop,
            chk_only_t_s,
            chk_clean
        ],
        outputs=[plot_output, df_output]
    )

if __name__ == "__main__":
    demo.launch()
