import torch


class OutputComparison:

    @staticmethod
    def get_probabilities(logits):
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def get_top_token(model, logits):
        final_logits = logits[0, -1]
        token_id = final_logits.argmax().item()

        return model.tokenizer.decode([token_id])

    @staticmethod
    def get_top_probability(logits):
        probabilities = OutputComparison.get_probabilities(logits)

        return probabilities[0, -1].max().item()

    @staticmethod
    def get_logit_difference(
        original_logits,
        patched_logits
    ):
        original_final = original_logits[0, -1]
        patched_final = patched_logits[0, -1]

        difference = patched_final - original_final

        return difference

    @staticmethod
    def compare(
        model,
        original_logits,
        patched_logits
    ):
        original_token = OutputComparison.get_top_token(
            model,
            original_logits
        )

        patched_token = OutputComparison.get_top_token(
            model,
            patched_logits
        )

        original_probability = (
            OutputComparison.get_top_probability(
                original_logits
            )
        )

        patched_probability = (
            OutputComparison.get_top_probability(
                patched_logits
            )
        )

        logit_difference = (
            OutputComparison.get_logit_difference(
                original_logits,
                patched_logits
            )
        )

        return {
            "original_token": original_token,
            "patched_token": patched_token,
            "original_probability": original_probability,
            "patched_probability": patched_probability,
            "probability_change": (
                patched_probability - original_probability
            ),
            "maximum_logit_change": (
                logit_difference.abs().max().item()
            )
        }