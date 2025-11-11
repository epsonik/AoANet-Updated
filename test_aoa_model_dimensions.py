"""
Test to verify AoAModel correctly handles different attention feature sizes.
This test ensures the att_embed layer uses the correct dimensions.
"""
import torch
import argparse
import sys

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.AoAModel import AoAModel


def test_aoa_model_with_different_att_feat_sizes():
    """Test that AoAModel can handle different attention feature sizes."""
    
    # Test configurations with different att_feat_size values
    test_configs = [
        {'att_feat_size': 1024, 'rnn_size': 512},
        {'att_feat_size': 2048, 'rnn_size': 512},
        {'att_feat_size': 2208, 'rnn_size': 1024},  # This is the problematic case from the issue
    ]
    
    for config in test_configs:
        print(f"\nTesting with att_feat_size={config['att_feat_size']}, rnn_size={config['rnn_size']}")
        
        # Create options object
        opt = argparse.Namespace()
        opt.vocab_size = 10000
        opt.input_encoding_size = 512
        opt.rnn_size = config['rnn_size']
        opt.num_layers = 2
        opt.drop_prob_lm = 0.5
        opt.fc_feat_size = 2048
        opt.att_feat_size = config['att_feat_size']
        opt.att_hid_size = 512
        opt.use_bn = 0
        opt.vocab = {}
        opt.mean_feats = 1
        opt.use_multi_head = 2
        opt.multi_head_scale = 1
        opt.num_heads = 8
        opt.refine = 1
        opt.refine_aoa = 1
        opt.use_ff = 1
        opt.dropout_aoa = 0.3
        
        # Create model
        try:
            model = AoAModel(opt)
            print(f"✓ Model created successfully")
            
            # Check att_embed dimensions
            first_layer = model.att_embed[0]
            expected_in_features = config['att_feat_size']
            expected_out_features = config['rnn_size']
            
            assert first_layer.in_features == expected_in_features, \
                f"Expected att_embed input size {expected_in_features}, got {first_layer.in_features}"
            assert first_layer.out_features == expected_out_features, \
                f"Expected att_embed output size {expected_out_features}, got {first_layer.out_features}"
            
            print(f"✓ att_embed dimensions correct: {first_layer.in_features} -> {first_layer.out_features}")
            
            # Test forward pass with dummy data
            batch_size = 2
            seq_length = 196  # Common for 14x14 grid
            
            fc_feats = torch.randn(batch_size, opt.fc_feat_size)
            att_feats = torch.randn(batch_size, seq_length, config['att_feat_size'])
            
            # Test _prepare_feature method
            mean_feats, att_feats_out, p_att_feats, att_masks = model._prepare_feature(
                fc_feats, att_feats, None
            )
            
            print(f"✓ Forward pass through _prepare_feature successful")
            print(f"  mean_feats shape: {mean_feats.shape}")
            print(f"  att_feats_out shape: {att_feats_out.shape}")
            
            # Verify output dimensions
            assert mean_feats.shape == (batch_size, config['rnn_size']), \
                f"Expected mean_feats shape ({batch_size}, {config['rnn_size']}), got {mean_feats.shape}"
            assert att_feats_out.shape == (batch_size, seq_length, config['rnn_size']), \
                f"Expected att_feats_out shape ({batch_size}, {seq_length}, {config['rnn_size']}), got {att_feats_out.shape}"
            
            print(f"✓ Output dimensions correct")
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_aoa_model_with_different_att_feat_sizes()
    sys.exit(0 if success else 1)
