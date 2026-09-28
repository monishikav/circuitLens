import torch


class ActivationHook:
    def __init__(self):
        self.activations = {}

    def save_activation(self, name):
        def hook_fn(activation, hook):
            self.activations[name] = activation.detach().cpu()

        return hook_fn

    def clear(self):
        self.activations.clear()

    def get_activation(self, name):
        return self.activations.get(name)

    def get_all(self):
        return self.activations