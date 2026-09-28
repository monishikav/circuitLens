import torch

from src.circuitLens.hooks.activation_hooks import ActivationHook


def test_save_activation():
    hook = ActivationHook()

    activation = torch.tensor([[1.0, 2.0, 3.0]])

    hook_fn = hook.save_activation("test")

    hook_fn(activation, None)

    result = hook.get_activation("test")

    assert result is not None
    assert torch.equal(result, activation)


def test_clear_activation():
    hook = ActivationHook()

    hook.activations["test"] = torch.tensor([1.0])

    hook.clear()

    assert hook.get_all() == {}