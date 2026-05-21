import gradio as gr
import json
import pandas as pd
from src.utils import predict_labels, get_description
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
                dd_model_type = gr.Dropdown(
                    choices=[
                        ("TF-IDF Logistic Regression", "baseline"),
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
                    chk_weighted = gr.Checkbox(label="Weighted", interactive=True)
                    chk_thrs_tuned = gr.Checkbox(label="Threshold Tuned", interactive=True)

    btn_predict = gr.Button("Predict Labels", variant="primary")

    df_output = gr.Dataframe(
        headers=["Label", "Description"],
        datatype="str",
        interactive=False,
        wrap=True,
    )
    plot_output = gr.BarPlot(
        x="score",
        y="label",
        color="predicted",
        color_map={"True": "green", "False": "red"},
        tooltip=["label", "score", "predicted", "threshold", "description"],
    )

    with gr.Row():
        with gr.Column(scale=1):
            dd_heatmap_label = gr.Dropdown(
                choices=[],
                interactive=True,
                label=""
            )

        with gr.Column(scale=9):
            html_view = gr.HTML()

    stored_heatmaps = gr.State({})

    def toggle_config(config_value):

        if config_value == "custom":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)


    def update_heatmap_view(selected_label, heatmaps_state):
        if not selected_label or not heatmaps_state:
            return "<div></div>"
        return heatmaps_state.get(selected_label, "<div>Heatmap not found.</div>")


    def predict_labels_wrapper(
            tb_text: str,
            dd_task: str,
            dd_model_type: str,
            dd_config: str,
            chk_free_text: bool,
            dd_metric: str,
            chk_with_drop: bool,
            chk_only_t_s: bool,
            chk_clean: bool,
            chk_weighted: bool,
            chk_thrs_tuned: bool,
    ):

        if not tb_text.strip():
            raise gr.Error("No input text provided.")


        result = predict_labels(
            text=tb_text,
            task=dd_task,
            model_type=dd_model_type,
            config=dd_config,
            free_text=chk_free_text,
            metric=dd_metric,
            nlp_core=NLP_CORE,
            t_s_config_data=T_S_CONFIG_DATA,
            with_drop=chk_with_drop,
            only_t_s=chk_only_t_s,
            clean=chk_clean,
            weighted=chk_weighted,
            thrs_tuned=chk_thrs_tuned
        )


        if "error" in result:
            raise gr.Error(result["error"])

        predicted_labels = result.get("predicted_labels", [])
        labels_values = result.get("labels_n_values", {})

        table_data = []
        for label in predicted_labels:

            description = get_description(label, CHAPTER_DESCRIPTIONS, CODE_DESCRIPTIONS)

            table_data.append([label, description])

        df_table = pd.DataFrame(table_data, columns=["Label", "Description"])

        data = []

        for label, (score, threshold) in labels_values.items():

            predicted = "True" if label in predicted_labels else "False"

            description = get_description(label, CHAPTER_DESCRIPTIONS, CODE_DESCRIPTIONS)

            data.append({
                "label": label,
                "score": round(score, 3),
                "predicted": predicted,
                "threshold": round(threshold, 3),
                "description": description
            })

        df_plot = pd.DataFrame(data)

        heatmaps_dict = result["heatmaps"]

        heatmap_labels = list(heatmaps_dict.keys())

        default_heatmap_label = heatmap_labels[0]

        dropdown_update = gr.update(
            choices=heatmap_labels,
            value=default_heatmap_label,
            visible=True
        )

        initial_html = heatmaps_dict.get(default_heatmap_label, "<div>No heatmap available.</div>")

        return (
            gr.update(value=df_plot),
            gr.update(value=df_table),
            heatmaps_dict,
            dropdown_update,
            gr.update(value=initial_html)
        )


    dd_heatmap_label.change(
        fn=update_heatmap_view,
        inputs=[dd_heatmap_label, stored_heatmaps],
        outputs=[html_view]
    )

    dd_config.change(
        fn=toggle_config,
        inputs=dd_config,
        outputs=[group_metric, group_custom]
    )

    btn_predict.click(
        fn=predict_labels_wrapper,
        inputs=[
            tb_text, dd_task, dd_model_type, dd_config, chk_free_text,
            dd_metric, chk_with_drop, chk_only_t_s, chk_clean, chk_weighted,
            chk_thrs_tuned
        ],
        outputs=[plot_output, df_output, stored_heatmaps, dd_heatmap_label, html_view],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.queue().launch()
