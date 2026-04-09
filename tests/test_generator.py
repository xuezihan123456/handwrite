import pytest
import torch
from torch import nn

from handwrite.engine.generator import Generator


def test_generator_instantiates_with_default_style_embedding():
    model = Generator()

    assert isinstance(model, Generator)
    assert model.style_embedding.num_embeddings == 5
    assert model.style_embedding.embedding_dim == 128


def test_generator_forward_returns_image_sized_output():
    model = Generator(num_styles=5, embed_dim=128)
    inputs = torch.randn(2, 1, 256, 256)
    style_ids = torch.tensor([0, 4], dtype=torch.long)

    outputs = model(inputs, style_ids)

    assert outputs.shape == (2, 1, 256, 256)


def test_generator_output_is_bounded_by_tanh():
    model = Generator(num_styles=5, embed_dim=128)
    inputs = torch.randn(1, 1, 256, 256)
    style_ids = torch.tensor([2], dtype=torch.long)

    outputs = model(inputs, style_ids)

    assert torch.all(outputs >= -1.0)
    assert torch.all(outputs <= 1.0)


@pytest.mark.parametrize(
    ("images", "style_ids"),
    [
        (torch.randn(256, 256), torch.tensor([0], dtype=torch.long)),
        (torch.randn(2, 3, 256, 256), torch.tensor([0, 1], dtype=torch.long)),
        (torch.randn(2, 1, 128, 256), torch.tensor([0, 1], dtype=torch.long)),
    ],
)
def test_generator_rejects_invalid_image_shapes(images, style_ids):
    model = Generator(num_styles=5, embed_dim=128)

    with pytest.raises(ValueError, match=r"images must have shape \(batch, 1, 256, 256\)"):
        model(images, style_ids)


@pytest.mark.parametrize(
    "style_ids",
    [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    ],
)
def test_generator_rejects_invalid_style_id_shapes(style_ids):
    model = Generator(num_styles=5, embed_dim=128)
    images = torch.randn(2, 1, 256, 256)

    with pytest.raises(
        ValueError, match=r"style_ids must have shape \(batch,\) matching the image batch size"
    ):
        model(images, style_ids)


def test_generator_changes_output_when_style_id_changes():
    torch.manual_seed(0)
    model = Generator(num_styles=5, embed_dim=128).eval()
    inputs = torch.zeros(1, 1, 256, 256)

    with torch.no_grad():
        output_style_zero = model(inputs, torch.tensor([0], dtype=torch.long))
        output_style_one = model(inputs, torch.tensor([1], dtype=torch.long))

    difference = torch.max(torch.abs(output_style_zero - output_style_one)).item()

    assert difference > 1e-5


def test_generator_matches_documented_unet_topology_and_scale():
    model = Generator(num_styles=5, embed_dim=128)

    assert isinstance(model.e1.block[0], nn.Conv2d)
    assert model.e1.block[0].in_channels == 1
    assert model.e1.block[0].out_channels == 64
    assert isinstance(model.e8.block[0], nn.Conv2d)
    assert model.e8.block[0].out_channels == 512
    assert isinstance(model.d1.block[0], nn.ConvTranspose2d)
    assert model.d1.block[0].in_channels == 640
    assert model.d2.block[0].in_channels == 1024
    assert model.d5.block[0].out_channels == 256
    assert isinstance(model.d8.block[0], nn.ConvTranspose2d)
    assert model.d8.block[0].out_channels == 1
    assert isinstance(model.d8.block[-1], nn.Tanh)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 55_000_000 <= parameter_count <= 56_000_000
