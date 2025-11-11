"""
Simple test script to verify visualization utilities work correctly.
This tests the individual components without requiring a trained model.
"""
import numpy as np
import torch
from PIL import Image
import os
import sys

# Import visualization utilities
import vis_utils

def test_resize_attention():
    """Test attention map resizing."""
    print("Testing resize_attention_to_image...")
    
    # Create dummy attention weights for a 7x7 grid (49 regions)
    attention = np.random.rand(49)
    attention = attention / attention.sum()  # Normalize
    
    # Resize to image dimensions
    image_shape = (224, 224)
    attention_map = vis_utils.resize_attention_to_image(attention, image_shape, 49)
    
    assert attention_map.shape == image_shape, f"Expected shape {image_shape}, got {attention_map.shape}"
    assert attention_map.min() >= 0 and attention_map.max() <= 1, "Attention map should be normalized to [0, 1]"
    
    print("✓ Resize attention test passed")
    return True

def test_create_heatmap():
    """Test heatmap creation."""
    print("Testing create_attention_heatmap...")
    
    # Create a dummy image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create dummy attention map
    attention_map = np.random.rand(224, 224)
    
    # Create heatmap
    heatmap = vis_utils.create_attention_heatmap(image, attention_map)
    
    assert heatmap.shape == image.shape, f"Heatmap shape should match image shape"
    assert heatmap.dtype == np.uint8, "Heatmap should be uint8"
    
    print("✓ Create heatmap test passed")
    return True

def test_visualize_attention():
    """Test full attention visualization pipeline."""
    print("Testing visualize_attention_for_sequence...")
    
    # Use the sample image
    image_path = './vis/COCO_test2014_000000000027.jpg'
    
    if not os.path.exists(image_path):
        print(f"Warning: Sample image not found at {image_path}, skipping test")
        return True
    
    # Create dummy attention weights (5 words, 49 attention regions each)
    num_words = 5
    att_size = 49
    attention_weights = [torch.rand(1, att_size) for _ in range(num_words)]
    word_sequence = ['a', 'cat', 'on', 'the', 'couch']
    
    # Create visualizations
    output_dir = '/tmp/test_attention_vis'
    os.makedirs(output_dir, exist_ok=True)
    
    vis_paths = vis_utils.visualize_attention_for_sequence(
        image_path,
        attention_weights,
        word_sequence,
        output_dir=output_dir,
        att_size=att_size
    )
    
    assert len(vis_paths) == num_words, f"Expected {num_words} visualizations, got {len(vis_paths)}"
    
    # Check that files were created
    for path in vis_paths:
        assert os.path.exists(path), f"Visualization file not created: {path}"
    
    # Check summary was created
    summary_path = os.path.join(output_dir, 'COCO_test2014_000000000027_summary.png')
    assert os.path.exists(summary_path), f"Summary visualization not created: {summary_path}"
    
    print(f"✓ Visualize attention test passed. Created {len(vis_paths)} visualizations")
    print(f"  Output directory: {output_dir}")
    
    return True

def test_attention_hook():
    """Test attention hook class."""
    print("Testing AttentionHook...")
    
    # Create a dummy module with attn attribute
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = None
    
    module = DummyModule()
    hook = vis_utils.AttentionHook()
    
    # Simulate attention computation
    module.attn = torch.rand(1, 8, 49)  # batch_size=1, num_heads=8, att_size=49
    
    # Call hook
    hook(module, None, None)
    
    assert len(hook.attention_weights) == 1, "Hook should capture one attention weight"
    assert hook.attention_weights[0].shape == (1, 49), "Hook should average across heads"
    
    # Test reset
    hook.reset()
    assert len(hook.attention_weights) == 0, "Reset should clear attention weights"
    
    print("✓ Attention hook test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Attention Visualization Utilities")
    print("=" * 60)
    print()
    
    tests = [
        test_resize_attention,
        test_create_heatmap,
        test_attention_hook,
        test_visualize_attention,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
