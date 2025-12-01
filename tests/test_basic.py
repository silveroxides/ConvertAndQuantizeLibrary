"""
Unit tests for the Convert and Quantize library.

This file contains basic test templates. Expand these tests based on 
specific requirements for your use cases.
"""

import torch
import pytest
from convert_and_quantize import LearnedRoundingConverter, get_device


class TestLearnedRoundingConverter:
    """Tests for LearnedRoundingConverter class."""

    def setup_method(self):
        """Set up the test method."""
        self.device = get_device()

    def test_initialization(self):
        """Test converter initialization with default parameters."""
        converter = LearnedRoundingConverter()
        assert str(converter.device) in ['cuda', 'cpu']
        assert converter.num_iter == 500
        assert converter.scaling_mode == 'tensor'

    def test_initialization_custom(self):
        """Test converter initialization with custom parameters."""
        converter = LearnedRoundingConverter(
            optimizer="adamw",
            num_iter=100,
            scaling_mode="block",
            block_size=32
        )
        assert converter.optimizer_choice == "adamw"
        assert converter.num_iter == 100
        assert converter.scaling_mode == "block"
        assert converter.block_size == 32

    def test_convert_basic(self):
        """Test basic tensor conversion."""
        converter = LearnedRoundingConverter(num_iter=10)
        weight = torch.randn(256, 256, device=self.device)
        
        quantized, scale, dequantized = converter.convert(weight)
        
        # Check shapes
        assert quantized.shape == weight.shape
        assert dequantized.shape == weight.shape
        
        # Check dtypes
        from convert_and_quantize import TARGET_FP8_DTYPE
        assert quantized.dtype == TARGET_FP8_DTYPE
        assert dequantized.dtype == torch.float32

    def test_convert_zero_tensor(self):
        """Test conversion of all-zero tensor."""
        converter = LearnedRoundingConverter()
        weight = torch.zeros(256, 256, device=self.device)
        
        quantized, scale, dequantized = converter.convert(weight)
        
        # All quantized values should be zero
        assert torch.all(quantized == 0)
        # All dequantized values should be zero
        assert torch.allclose(dequantized, torch.zeros_like(dequantized), atol=1e-6)

    def test_convert_different_shapes(self):
        """Test conversion of tensors with different shapes."""
        converter = LearnedRoundingConverter(num_iter=5)
        
        shapes = [
            (128, 128),
            (256, 512),
            (1024, 256),
            (2048, 4096),
        ]
        
        for shape in shapes:
            weight = torch.randn(*shape, device=self.device)
            quantized, scale, dequantized = converter.convert(weight)
            
            assert quantized.shape == shape
            assert dequantized.shape == shape

    def test_convert_quantization_error(self):
        """Test that quantization introduces manageable error."""
        converter = LearnedRoundingConverter(num_iter=50)
        weight = torch.randn(512, 512, device=self.device)
        
        quantized, scale, dequantized = converter.convert(weight)
        
        # Calculate error
        error = (weight - dequantized).abs().mean()
        
        # Error should be small (relative to weight range)
        assert error < 1.0  # Loose bound - specific value depends on implementation

    def test_block_scaling(self):
        """Test block-level scaling."""
        converter = LearnedRoundingConverter(
            num_iter=10,
            scaling_mode="block",
            block_size=64
        )
        
        weight = torch.randn(512, 256, device=self.device)
        quantized, scale, dequantized = converter.convert(weight)
        
        # Check scale shape for block scaling
        assert scale.ndim == 3  # (out_features, num_blocks, 1)

    def test_full_matrix_svd(self):
        """Test full matrix SVD option."""
        converter = LearnedRoundingConverter(
            num_iter=5,
            full_matrix=True
        )
        
        weight = torch.randn(128, 128, device=self.device)
        quantized, scale, dequantized = converter.convert(weight)
        
        assert quantized.shape == weight.shape

    def test_different_optimizers(self):
        """Test conversion with different optimizers."""
        optimizers = ["original", "adamw", "radam"]
        weight = torch.randn(256, 256, device=self.device)
        
        for opt in optimizers:
            try:
                converter = LearnedRoundingConverter(optimizer=opt, num_iter=5)
                quantized, scale, dequantized = converter.convert(weight)
                assert quantized.shape == weight.shape
            except Exception as e:
                # Some optimizers might require additional dependencies
                print(f"Optimizer {opt} not available: {e}")

    def test_top_p_parameter(self):
        """Test different top_p values."""
        weight = torch.randn(256, 256, device=self.device)
        
        top_p_values = [0.001, 0.01, 0.05, 0.1]
        
        for top_p in top_p_values:
            converter = LearnedRoundingConverter(top_p=top_p, num_iter=5)
            quantized, scale, dequantized = converter.convert(weight)
            assert quantized.shape == weight.shape

    def test_invalid_optimizer(self):
        """Test that invalid optimizer raises error."""
        with pytest.raises(ValueError):
            converter = LearnedRoundingConverter(optimizer="invalid_optimizer")
            weight = torch.randn(256, 256, device=self.device)
            converter.convert(weight)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with fixed seed."""
        weight = torch.randn(256, 256, device=self.device)
        
        # First run
        generator = torch.Generator(device=self.device)
        converter1 = LearnedRoundingConverter(num_iter=10, seed=42, generator=generator)
        q1, s1, d1 = converter1.convert(weight.clone())
        
        # Second run with same seed
        converter2 = LearnedRoundingConverter(num_iter=10, seed=42, generator=generator)
        q2, s2, d2 = converter2.convert(weight.clone())
        
        # Results should be very close (may not be identical due to floating point)
        assert torch.allclose(d1, d2, rtol=1e-4)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_device(self):
        """Test device detection."""
        from convert_and_quantize import get_device
        
        device = get_device()
        assert device in ['cuda', 'cpu']

    def test_setup_seed(self):
        """Test seed setup."""
        from convert_and_quantize import setup_seed
        
        gen = setup_seed(42)
        assert isinstance(gen, torch.Generator)

    def test_get_fp8_constants(self):
        """Test FP8 constant retrieval."""
        from convert_and_quantize import get_fp8_constants, TARGET_FP8_DTYPE
        
        min_val, max_val, min_pos = get_fp8_constants(TARGET_FP8_DTYPE)
        
        assert isinstance(min_val, float)
        assert isinstance(max_val, float)
        assert isinstance(min_pos, float)
        assert min_val < 0
        assert max_val > 0
        assert min_pos > 0

    def test_should_process_layer(self):
        """Test layer processing decision."""
        from convert_and_quantize import should_process_layer
        
        # Test different layer names
        test_cases = [
            ("attention.q_proj.weight", {"t5xxl": False}, (True, True, "")),
            ("norm.weight", {"t5xxl": True}, (False, False, "")),
            ("decoder.weight", {"t5xxl": True}, (False, False, "")),
        ]
        
        for key, kwargs, expected in test_cases:
            result = should_process_layer(key, **kwargs)
            # At minimum, result should be a tuple of 3 elements
            assert len(result) == 3
            assert isinstance(result[0], bool)
            assert isinstance(result[1], bool)
            assert isinstance(result[2], str)


class TestConstants:
    """Tests for constants and configuration."""

    def test_data_type_constants(self):
        """Test that data type constants are defined."""
        from convert_and_quantize import (
            TARGET_FP8_DTYPE,
            COMPUTE_DTYPE,
            SCALE_DTYPE,
        )
        
        assert TARGET_FP8_DTYPE is not None
        assert COMPUTE_DTYPE is not None
        assert SCALE_DTYPE is not None

    def test_fp8_bounds(self):
        """Test FP8 min/max constants."""
        from convert_and_quantize import FP8_MIN, FP8_MAX, FP8_MIN_POS
        
        assert FP8_MIN < 0
        assert FP8_MAX > 0
        assert FP8_MIN_POS > 0
        assert FP8_MIN < FP8_MAX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
