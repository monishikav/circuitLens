import torch


class AttentionMetrics:

    @staticmethod
    def attention_entropy(attention):
        """
        Calculates entropy of attention distribution.

        Expected input shape:
        [layers, batch, heads, query_tokens, key_tokens]
        """

        attention = attention.float()

        # Avoid log(0)
        attention = torch.clamp(attention, min=1e-12)

        entropy = -torch.sum(
            attention * torch.log(attention),
            dim=-1
        )

        return entropy

    @staticmethod
    def maximum_attention(attention):
        """
        Finds the strongest attention value for each
        layer, head and query token.
        """

        return torch.max(attention, dim=-1).values

    @staticmethod
    def average_attention(attention):
        """
        Calculates the average attention value.
        """

        return torch.mean(attention, dim=-1)

    @staticmethod
    def summarize(attention):
        """
        Creates a compact summary of attention metrics.
        """

        entropy = AttentionMetrics.attention_entropy(attention)
        maximum = AttentionMetrics.maximum_attention(attention)
        average = AttentionMetrics.average_attention(attention)

        return {
            "entropy": entropy,
            "maximum": maximum,
            "average": average
        }