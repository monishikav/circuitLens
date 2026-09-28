import json
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.circuitLens.models.loader import ModelLoader
from src.circuitLens.attention.analyzer import AttentionAnalyzer
from src.circuitLens.patching.activation_patch import ActivationPatcher
from src.circuitLens.analysis.comparison import OutputComparison


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.circuitLens.models.loader import ModelLoader
from src.circuitLens.attention.analyzer import AttentionAnalyzer
from src.circuitLens.attention.analyzer import AttentionAnalyzer


st.set_page_config(
    page_title="CircuitLens",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 CircuitLens")
st.subheader("Transformer Interpretability Dashboard")


@st.cache_resource
def load_model():
    loader = ModelLoader()
    return loader.load_model()


def load_results():
    with open(
        "experiments/results/experiment_results.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


model = load_model()
analyzer = AttentionAnalyzer(model)

results = load_results()

st.write(f"Experiments available: {len(results)}")

st.divider()

st.subheader("Attention Explorer")

prompt_options = [
    result["prompt"]
    for result in results
]

selected_prompt = st.selectbox(
    "Select a prompt",
    prompt_options
)

layer = st.slider(
    "Select Transformer Layer",
    min_value=0,
    max_value=model.cfg.n_layers - 1,
    value=5
)

head = st.slider(
    "Select Attention Head",
    min_value=0,
    max_value=model.cfg.n_heads - 1,
    value=0
)

if st.button("Analyze Attention"):

    heatmap_data = analyzer.get_heatmap_data(
        selected_prompt,
        layer,
        head
    )

    tokens = heatmap_data["tokens"]
    attention = heatmap_data["attention"].numpy()

    figure = px.imshow(
        attention,
        x=tokens,
        y=tokens,
        text_auto=".2f",
        labels={
            "x": "Key Token",
            "y": "Query Token",
            "color": "Attention"
        },
        title=f"Layer {layer} — Head {head}"
    )

    figure.update_layout(
        height=600
    )

    st.plotly_chart(
        figure,
        use_container_width=True
    )
    st.divider()

st.subheader("Attention Head Comparison")

if st.button("Compare All Heads"):

    head_summary = analyzer.get_head_summary(
        selected_prompt
    )

    st.write(
        f"Analyzing {len(head_summary)} attention heads..."
    )

    for item in head_summary:
        item["head_name"] = (
            f"Layer {item['layer']} - "
            f"Head {item['head']}"
        )

    selected_layer = st.selectbox(
        "Select layer for comparison",
        range(model.cfg.n_layers),
        key="comparison_layer"
    )

    layer_results = [
        item
        for item in head_summary
        if item["layer"] == selected_layer
    ]

    layer_results = sorted(
        layer_results,
        key=lambda x: x["average_entropy"]
    )

    st.dataframe(
        layer_results,
        use_container_width=True
    )
    st.divider()

st.subheader("Activation Patching")

source_prompt = st.text_input(
    "Source prompt",
    value="The capital of France is"
)

target_prompt = st.text_input(
    "Target prompt",
    value="The capital of Germany is"
)

patch_layer = st.slider(
    "Patching layer",
    min_value=0,
    max_value=model.cfg.n_layers - 1,
    value=5,
    key="patch_layer"
)

patch_position = st.number_input(
    "Token position",
    min_value=-10,
    max_value=10,
    value=-1,
    step=1
)

if st.button("Run Activation Patch",key="run_activation_patch"):

    patcher = ActivationPatcher(model)

    patch_result = patcher.patch_activation(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        layer=patch_layer,
        position=patch_position
    )

    comparison = OutputComparison.compare(
        model,
        patch_result["original_logits"],
        patch_result["patched_logits"]
    )

    st.success("Activation patching completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Token",
            comparison["original_token"]
        )

        st.metric(
            "Original Probability",
            f"{comparison['original_probability']:.4f}"
        )

    with col2:
        st.metric(
            "Patched Token",
            comparison["patched_token"]
        )

        st.metric(
            "Patched Probability",
            f"{comparison['patched_probability']:.4f}"
        )

    st.metric(
        "Probability Change",
        f"{comparison['probability_change']:.4f}"
    )

    st.metric(
        "Maximum Logit Change",
        f"{comparison['maximum_logit_change']:.4f}"
    )
    st.divider()

st.subheader("Activation Patching")

source_prompt = st.text_input(
    "Source Prompt",
    value="The capital of France is"
)

target_prompt = st.text_input(
    "Target Prompt",
    value="The capital of Germany is"
)

patch_layer = st.slider(
    "Patching Layer",
    min_value=0,
    max_value=model.cfg.n_layers - 1,
    value=5
)

patch_position = st.number_input(
    "Token Position",
    min_value=-10,
    max_value=10,
    value=-1,
    step=1
)

if st.button("Run Activation Patch"):

    patcher = ActivationPatcher(model)

    result = patcher.patch_activation(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        layer=patch_layer,
        position=patch_position
    )

    comparison = OutputComparison.compare(
        model,
        result["original_logits"],
        result["patched_logits"]
    )

    st.success("Activation patching completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Token",
            comparison["original_token"]
        )

        st.metric(
            "Original Probability",
            f"{comparison['original_probability']:.4f}"
        )

    with col2:
        st.metric(
            "Patched Token",
            comparison["patched_token"]
        )

        st.metric(
            "Patched Probability",
            f"{comparison['patched_probability']:.4f}"
        )

    st.metric(
        "Probability Change",
        f"{comparison['probability_change']:.4f}"
    )

    st.metric(
        "Maximum Logit Change",
        f"{comparison['maximum_logit_change']:.4f}"
    )