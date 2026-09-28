from src.circuitLens.models.loader import ModelLoader
from src.circuitLens.patching.activation_patch import ActivationPatcher
from src.circuitLens.analysis.comparison import OutputComparison
from src.circuitLens.storage.experiment_store import ExperimentStore


def main():
    loader = ModelLoader()
    model = loader.load_model()

    patcher = ActivationPatcher(model)
    store = ExperimentStore()

    source_prompt = "The capital of France is"
    target_prompt = "The capital of Germany is"

    layer = 5

    print("Source prompt:", source_prompt)
    print("Target prompt:", target_prompt)
    print("Patching layer:", layer)

    result = patcher.patch_activation(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        layer=layer,
        position=-1
    )

    comparison = OutputComparison.compare(
        model,
        result["original_logits"],
        result["patched_logits"]
    )

    print("\n" + "=" * 50)
    print("ACTIVATION PATCHING COMPARISON")
    print("=" * 50)

    print("Original top token:",
          repr(comparison["original_token"]))

    print("Patched top token:",
          repr(comparison["patched_token"]))

    print("Original probability:",
          comparison["original_probability"])

    print("Patched probability:",
          comparison["patched_probability"])

    print("Probability change:",
          comparison["probability_change"])

    print("Maximum logit change:",
          comparison["maximum_logit_change"])

    patching_result = {
        "experiment_type": "activation_patching",
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "layer": layer,
        "position": -1,
        **comparison
    }

    store.save_results(
        [patching_result],
        filename="patching_results.json"
    )


if __name__ == "__main__":
    main()