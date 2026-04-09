import torch

from handwrite.engine.discriminator import Discriminator


def test_discriminator_can_be_instantiated() -> None:
    model = Discriminator(num_styles=5)

    assert isinstance(model, Discriminator)


def test_discriminator_forward_returns_patch_outputs_with_style_logits() -> None:
    model = Discriminator(num_styles=5)
    inputs = torch.randn(8, 1, 256, 256)

    real_fake, category = model(inputs)

    assert real_fake.shape == (8, 1, 30, 30)
    assert category.shape == (8, 5, 30, 30)


def test_discriminator_parameter_count_stays_in_expected_range() -> None:
    model = Discriminator(num_styles=5)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 2_500_000 <= parameter_count <= 6_000_000
