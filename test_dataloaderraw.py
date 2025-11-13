"""
Test script to verify dataloaderraw.py handles different CNN models correctly.
"""
import sys
import os

def test_dataloader_initialization():
    """Test that DataLoaderRaw initializes with correct feature sizes for different models."""
    sys.path.append('/home/runner/work/AoANet-Updated/AoANet-Updated')
    
    from dataloaderraw import DataLoaderRaw
    
    test_cases = [
        ('densenet121', 1024),
        ('densenet161', 2208),
        ('densenet169', 1664),
        ('densenet201', 1920),
        ('resnet152', 2048),  # This should work after our fix
        ('inception', 2048),
    ]
    
    print("Testing DataLoaderRaw initialization with different CNN models...")
    print("=" * 70)
    
    # Create a dummy folder for testing (won't actually use it)
    test_folder = '/tmp/test_images'
    os.makedirs(test_folder, exist_ok=True)
    
    for cnn_model, expected_size in test_cases:
        print(f"\nTesting {cnn_model}...")
        try:
            loader = DataLoaderRaw({
                'folder_path': test_folder,
                'coco_json': '',
                'batch_size': 1,
                'cnn_model': cnn_model
            })
            
            actual_size = loader.feature_size
            
            if actual_size == expected_size:
                print(f"  ✓ PASS: {cnn_model} - Expected size: {expected_size}, Got: {actual_size}")
            else:
                print(f"  ✗ FAIL: {cnn_model} - Expected size: {expected_size}, Got: {actual_size}")
                
        except Exception as e:
            print(f"  ✗ ERROR: {cnn_model} - {str(e)}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    test_dataloader_initialization()
