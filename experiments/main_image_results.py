import json
import plotly.graph_objects as go

filenames = {
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_feedback_aware_sft_smart_duplicates.json": "Feedback-Aware (ours)",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_single_answer_sft_smart_duplicates.json": "Single Selection",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_answers_sft_smart_duplicates.json": "All Selections",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_answers_with_aid_sft_smart_duplicates.json": "All Selections w/ <br>Generated Content",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_all_gpt_one_shot_final.json": "GPT-4o-mini One Shot",
}

fig_precision = go.Figure()
fig_mean = go.Figure()
fig_mean_good = go.Figure()

for filename in filenames:
    print(f"Processing {filename}")
    dataset = json.load(open(filename, 'r'))        

    precision_at_k = []
    precision_at_k_repeated = []
    mean_prob_qa_at_k = []
    mean_prog_qa_good_at_k = []

    for data in dataset:
        if len(data) == 0:
            continue

        precision_at_k_dict = {}
        precision_at_k_repeated_dict = {}
        mean_prob_qa_at_k_dict = {}
        mean_prog_qa_good_at_k_dict = {}

        for i in range(1, 26):
            precision_at_k_dict[i] = len([1 for x in data[:i] if x['label'] == 'GOOD']) / i
            precision_at_k_repeated_dict[f"{i}"] = len([1 for x in data[:i] if x['label'] == 'REPEATED']) / i
            mean_prob_qa_at_k_dict[f"mean_prob_qa_at_{i}"] = sum([x['prob_qa'] for x in data[:i]]) / i
            mean_prog_qa_good_at_k_dict[f"mean_prog_qa_good_at_{i}"] = sum([x['prob_qa'] for x in data[:i] if x['label'] == 'GOOD']) / i

        precision_at_k.append(precision_at_k_dict)
        precision_at_k_repeated.append(precision_at_k_repeated_dict)
        mean_prob_qa_at_k.append(mean_prob_qa_at_k_dict)
        mean_prog_qa_good_at_k.append(mean_prog_qa_good_at_k_dict)

    mean_precision_at_k_dict = {}
    mean_precision_at_k_repeated_dict = {}
    mean_mean_prob_qa_at_k_dict = {}
    mean_mean_prog_qa_good_at_k_dict = {}

    for i in range(1, 26):
        mean_precision_at_k_dict[i] = sum([x[i] for x in precision_at_k]) / len(precision_at_k)
        mean_precision_at_k_repeated_dict[f"{i}"] = sum([x[f"{i}"] for x in precision_at_k_repeated]) / len(precision_at_k_repeated)
        mean_mean_prob_qa_at_k_dict[f"mean_prob_qa_at_{i}"] = sum([x[f"mean_prob_qa_at_{i}"] for x in mean_prob_qa_at_k]) / len(mean_prob_qa_at_k)
        mean_mean_prog_qa_good_at_k_dict[f"mean_prog_qa_good_at_{i}"] = sum([x[f"mean_prog_qa_good_at_{i}"] for x in mean_prog_qa_good_at_k]) / len(mean_prog_qa_good_at_k)

    fig_mean.add_trace(go.Scatter(x=list(mean_mean_prob_qa_at_k_dict.keys()), y=list(mean_mean_prob_qa_at_k_dict.values()), mode='lines+markers', name=filenames[filename]))
    fig_precision.add_trace(go.Scatter(x=list(mean_precision_at_k_dict.keys()), y=list(mean_precision_at_k_dict.values()), mode='lines+markers', name=filenames[filename]))
    fig_mean_good.add_trace(go.Scatter(x=list(mean_mean_prog_qa_good_at_k_dict.keys()), y=list(mean_mean_prog_qa_good_at_k_dict.values()), mode='lines+markers', name=filenames[filename]))

fig_precision.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="K",
    yaxis_title="Precision",
    width=1000,
    height=600,
    xaxis=dict(tickfont=dict(size=14)),
    yaxis=dict(tickfont=dict(size=14)),
    legend=dict(
        x=0.02,  # Position the legend inside, adjust as needed
        y=0.02,
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.5)",  # Optional: Add background transparency
        bordercolor="Black",
        borderwidth=2
    ),
)
fig_precision.write_image("Feedback-Aware-Questions/experiments/precision_at_k_main.png")

fig_mean.update_layout(
    title="Mean QA Loss at K",
    xaxis_title="K",
    yaxis_title="Mean QA Loss",
    width=1000,
    height=600,
)
fig_mean.write_image("Feedback-Aware-Questions/experiments/mean_prob_qa_at_k.png")

fig_mean_good.update_layout(
    title="Mean QA Loss for Good at K",
    xaxis_title="K",
    yaxis_title="Mean QA Loss",
    width=1000,
    height=600,
)
fig_mean_good.write_image("Feedback-Aware-Questions/experiments/mean_prog_qa_good_at_k.png")

