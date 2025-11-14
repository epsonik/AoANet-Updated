"""
Test script to verify that heatmaps are organized in per-image subdirectories.
This tests the integration of the directory structure changes.
"""
import os
import sys
import shutil

def test_demo_directory_structure():
    """Test that demo_visualization creates subdirectories per image."""
    print("Testing demo visualization directory structure...")
    
    # Clean up any existing output
    output_dir = './vis/demo_attention'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Run the demo
    import demo_visualization
    try:
        demo_visualization.main()
    except Exception as e:
        print(f"✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check that the subdirectory was created
    image_basename = 'COCO_test2014_000000000027'
    image_subdir = os.path.join(output_dir, image_basename)
    
    if not os.path.exists(image_subdir):
        print(f"✗ Image subdirectory not created: {image_subdir}")
        return False
    
    # Check that heatmap files are in the subdirectory
    files_in_subdir = os.listdir(image_subdir)
    heatmap_files = [f for f in files_in_subdir if f.endswith('.png') and not f.endswith('_summary.png')]
    
    if len(heatmap_files) == 0:
        print(f"✗ No heatmap files found in subdirectory: {image_subdir}")
        return False
    
    # Check that the original image is present
    original_image = os.path.join(image_subdir, 'original.jpg')
    if not os.path.exists(original_image):
        print(f"✗ Original image not found in subdirectory: {original_image}")
        return False
    
    print(f"✓ Demo directory structure test passed")
    print(f"  Subdirectory created: {image_subdir}")
    print(f"  Found {len(heatmap_files)} heatmap files in subdirectory")
    print(f"  Original image present: original.jpg")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Heatmap Directory Structure")
    print("=" * 60)
    print()
    
    # Check if sample image exists
    sample_image = './vis/COCO_test2014_000000000027.jpg'
    if not os.path.exists(sample_image):
        print(f"Error: Sample image not found at {sample_image}")
        print("Skipping directory structure tests.")
        return True
    
    tests = [
        test_demo_directory_structure,
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
