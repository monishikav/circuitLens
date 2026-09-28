import torch


class ActivationPatcher:
    def __init__(self, model):
        self.model = model

    def get_activation(self, prompt, layer):
        tokens = self.model.to_tokens(prompt)

        _, cache = self.model.run_with_cache(tokens)

        hook_name = f"blocks.{layer}.hook_resid_post"

        return tokens, cache[hook_name].detach()

    def patch_activation(
        self,
        source_prompt,
        target_prompt,
        layer,
        position=-1
    ):
        source_tokens, source_activation = self.get_activation(
            source_prompt,
            layer
        )

        target_tokens = self.model.to_tokens(target_prompt)

        hook_name = f"blocks.{layer}.hook_resid_post"

        def patch_hook(activation, hook):
            patched_activation = activation.clone()

            source_position = position

            patched_activation[:, position, :] = (
                source_activation[:, source_position, :]
                .to(activation.device)
            )

            return patched_activation

        original_logits = self.model(
            target_tokens
        )

        patched_logits = self.model.run_with_hooks(
            target_tokens,
            fwd_hooks=[
                (hook_name, patch_hook)
            ]
        )

        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "original_logits": original_logits.detach(),
            "patched_logits": patched_logits.detach(),
            "layer": layer,
            "position": position
        }