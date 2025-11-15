"""
Test script for inference.py functionality.
Tests the helper functions without requiring full model setup.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_extract_image_id():
    """Test image ID extraction from filename."""
    import re
    
    # Copy the function from inference.py
    def extract_image_id(filename):
        match = re.search(r'COCO_val2014_(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    print("Testing extract_image_id()...")
    
    # Test cases
    test_cases = [
        ("COCO_val2014_000000391895.jpg", 391895),
        ("COCO_val2014_000000123456.jpg", 123456),
        ("/path/to/COCO_val2014_000000000001.jpg", 1),
        ("COCO_val2014_000000999999.jpg", 999999),
        ("invalid_filename.jpg", None),
        ("other_format_123456.jpg", None),
    ]
    
    passed = 0
    failed = 0
    
    for filename, expected_id in test_cases:
        result = extract_image_id(filename)
        if result == expected_id:
            print(f"  ✓ {filename} -> {result}")
            passed += 1
        else:
            print(f"  ✗ {filename} -> {result} (expected {expected_id})")
            failed += 1
    
    print(f"\nImage ID extraction: {passed} passed, {failed} failed")
    return failed == 0


def test_load_reference_captions():
    """Test loading reference captions."""
    import json
    
    # Copy the function from inference.py
    def load_reference_captions(caption_file):
        print(f"Loading reference captions from {caption_file}...")
        with open(caption_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create mapping from image_id to captions
        image_to_captions = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in image_to_captions:
                image_to_captions[image_id] = []
            image_to_captions[image_id].append(caption)
        
        print(f"Loaded {len(image_to_captions)} images with reference captions")
        return image_to_captions
    
    print("\nTesting load_reference_captions()...")
    
    caption_file = "captions_val2014.json"
    
    if not os.path.exists(caption_file):
        print(f"  ⚠ Skipping test: {caption_file} not found")
        return True
    
    try:
        captions = load_reference_captions(caption_file)
        
        # Verify structure
        if not isinstance(captions, dict):
            print(f"  ✗ Expected dict, got {type(captions)}")
            return False
        
        if len(captions) == 0:
            print(f"  ✗ No captions loaded")
            return False
        
        # Check a sample entry
        sample_id = list(captions.keys())[0]
        sample_captions = captions[sample_id]
        
        if not isinstance(sample_captions, list):
            print(f"  ✗ Expected list of captions, got {type(sample_captions)}")
            return False
        
        print(f"  ✓ Loaded {len(captions)} images with reference captions")
        print(f"  ✓ Sample image {sample_id} has {len(sample_captions)} captions")
        print(f"    First caption: '{sample_captions[0][:60]}...'")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_argument_parser():
    """Test that argument parser is properly configured."""
    
    print("\nTesting argument parser...")
    
    # Check that the script has proper argument definitions
    try:
        parser_code = open('inference.py').read()
        
        required_args = [
            '--model',
            '--infos_path', 
            '--image_folder',
            '--reference_captions',
            '--beam_size',
            '--batch_size',
        ]
        
        missing = []
        for arg in required_args:
            if arg not in parser_code:
                missing.append(arg)
        
        if missing:
            print(f"  ✗ Missing arguments: {missing}")
            return False
        
        print(f"  ✓ All required arguments defined")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing inference.py")
    print("=" * 60)
    
    tests = [
        test_extract_image_id,
        test_load_reference_captions,
        test_argument_parser,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
