import torch


class AttentionAnalyzer:
    def __init__(self, model):
        self.model = model

    def get_attention(self, prompt):
        tokens = self.model.to_tokens(prompt)

        logits, cache = self.model.run_with_cache(tokens)

        attention = []

        for layer in range(self.model.cfg.n_layers):
            hook_name = f"blocks.{layer}.attn.hook_pattern"

            layer_attention = cache[hook_name]

            attention.append(
                layer_attention.detach().cpu()
            )

        attention = torch.stack(attention)

        return {
            "tokens": self.model.to_str_tokens(tokens),
            "attention": attention,
            "logits": logits.detach().cpu()
        }

    def get_attention_shape(self, prompt):
        result = self.get_attention(prompt)

        return result["attention"].shape

    def get_heatmap_data(self, prompt, layer, head):
        result = self.get_attention(prompt)

        attention = result["attention"]

        # Select one layer and one attention head
        heatmap = attention[
            layer,
            0,
            head
        ]

        return {
            "tokens": result["tokens"],
            "attention": heatmap
        }
    def get_head_summary(self, prompt):
        result = self.get_attention(prompt)

        attention = result["attention"]

        summaries = []

        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):

                head_attention = attention[
                    layer,
                    0,
                    head
                ]

                head_attention = torch.clamp(
                    head_attention.float(),
                    min=1e-12
                )

                entropy = -torch.sum(
                    head_attention * torch.log(head_attention),
                    dim=-1
                )

                summaries.append({
                    "layer": layer,
                    "head": head,
                    "average_entropy": entropy.mean().item()
                })

        return summaries