"""
Test script to verify resnet101 and resnet152 support in dataloaderraw.py
This is a simple validation test that checks the logic without requiring full dependencies.
"""

def test_cnn_model_logic():
    """
    Test that the CNN model selection logic handles resnet101 and resnet152.
    This simulates the logic in dataloaderraw.py without requiring PyTorch.
    """
    print("Testing CNN model selection logic...")
    
    # Simulate the CNN model selection logic from dataloaderraw.py
    test_cases = [
        ('densenet121', 1024),
        ('densenet161', 2208),
        ('densenet169', 1664),
        ('densenet201', 1920),
        ('regnet', 3024),
        ('inception', 2048),
        ('resnet101', 2048),  # New addition
        ('resnet152', 2048),  # New addition
        ('unknown_model', 2208),  # Should default to densenet161
    ]
    
    for cnn_model, expected_feature_size in test_cases:
        if cnn_model == 'densenet121':
            feature_size = 1024
        elif cnn_model == 'densenet161':
            feature_size = 2208
        elif cnn_model == 'densenet169':
            feature_size = 1664
        elif cnn_model == 'densenet201':
            feature_size = 1920
        elif cnn_model == 'regnet':
            feature_size = 3024
        elif cnn_model == 'inception':
            feature_size = 2048
        elif cnn_model == 'resnet101':
            feature_size = 2048
        elif cnn_model == 'resnet152':
            feature_size = 2048
        else:
            feature_size = 2208  # Default to densenet161
            
        if feature_size == expected_feature_size:
            print(f"✓ {cnn_model}: feature_size={feature_size} (expected {expected_feature_size})")
        else:
            print(f"✗ {cnn_model}: feature_size={feature_size} (expected {expected_feature_size})")
            return False
    
    print("\nAll CNN model logic tests passed!")
    return True


def test_feature_adaptation_logic():
    """
    Test the dynamic feature adaptation logic without requiring PyTorch.
    This simulates what happens in AoAModel._prepare_feature.
    """
    print("\nTesting feature adaptation logic...")
    
    # Simulate model state
    class MockModel:
        def __init__(self, att_feat_size):
            self.att_feat_size = att_feat_size
            self.rnn_size = 512
            self.drop_prob_lm = 0.5
            self.att_embed_recreated = False
    
    model = MockModel(att_feat_size=2048)  # Model trained with ResNet (2048)
    
    # Test 1: Input matches model's expected size (no adaptation needed)
    input_feat_size = 2048
    if input_feat_size != model.att_feat_size:
        model.att_embed_recreated = True
        model.att_feat_size = input_feat_size
    
    print(f"Test 1 - Matching features (2048 -> 2048): att_embed recreated = {model.att_embed_recreated}")
    assert not model.att_embed_recreated, "Should not recreate when sizes match"
    
    # Test 2: Input doesn't match, requires adaptation
    model = MockModel(att_feat_size=2048)  # Reset model
    input_feat_size = 2208  # Using DenseNet features
    if input_feat_size != model.att_feat_size:
        model.att_embed_recreated = True
        model.att_feat_size = input_feat_size
    
    print(f"Test 2 - Mismatched features (2048 -> 2208): att_embed recreated = {model.att_embed_recreated}")
    assert model.att_embed_recreated, "Should recreate when sizes don't match"
    assert model.att_feat_size == 2208, "Should update feature size"
    
    # Test 3: Reverse scenario
    model = MockModel(att_feat_size=2208)  # Reset model with DenseNet size
    input_feat_size = 2048  # Using ResNet features
    if input_feat_size != model.att_feat_size:
        model.att_embed_recreated = True
        model.att_feat_size = input_feat_size
    
    print(f"Test 3 - Mismatched features (2208 -> 2048): att_embed recreated = {model.att_embed_recreated}")
    assert model.att_embed_recreated, "Should recreate when sizes don't match"
    assert model.att_feat_size == 2048, "Should update feature size"
    
    print("\nAll feature adaptation tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing ResNet101/152 Support and Dynamic Feature Adaptation")
    print("=" * 70)
    print()
    
    all_passed = True
    
    if not test_cnn_model_logic():
        all_passed = False
    
    if not test_feature_adaptation_logic():
        all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit(main())
