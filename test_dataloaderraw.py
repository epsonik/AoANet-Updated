"""
Test script to verify dataloaderraw.py handles different CNN models correctly.
This test validates the code structure without requiring PyTorch or GPU.
"""
import ast
import sys

def test_dataloaderraw_code_structure():
    """
    Test that dataloaderraw.py has proper handling for resnet152.
    This is a static code analysis test that doesn't require PyTorch.
    """
    print("Testing dataloaderraw.py code structure...")
    print("=" * 70)
    
    # Read the dataloaderraw.py file
    with open('/home/runner/work/AoANet-Updated/AoANet-Updated/dataloaderraw.py', 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Find the DataLoaderRaw class
    dataloaderraw_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'DataLoaderRaw':
            dataloaderraw_class = node
            break
    
    assert dataloaderraw_class is not None, "DataLoaderRaw class not found"
    print("✓ Found DataLoaderRaw class")
    
    # Find the __init__ method
    init_method = None
    for item in dataloaderraw_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            init_method = item
            break
    
    assert init_method is not None, "__init__ method not found"
    print("✓ Found __init__ method")
    
    # Check if resnet152 is handled
    has_resnet152 = False
    has_correct_feature_size = False
    uses_myResnet = False
    
    init_source = ast.get_source_segment(content, init_method)
    
    if 'resnet152' in init_source:
        has_resnet152 = True
        print("✓ resnet152 handling found")
        
        # Check if feature size is set to 2048
        if "self.feature_size = 2048" in init_source:
            has_correct_feature_size = True
            print("✓ Feature size correctly set to 2048 for resnet152")
        
        # Check if myResnet is used
        if "myResnet" in init_source:
            uses_myResnet = True
            print("✓ myResnet wrapper is used for resnet152")
    
    # Check that myResnet is imported
    has_myResnet_import = "from misc.resnet_utils import myResnet" in content
    if has_myResnet_import:
        print("✓ myResnet is imported from misc.resnet_utils")
    
    # Check that self.my_cnn is used instead of self.my_densenet
    uses_my_cnn = "self.my_cnn" in content
    if uses_my_cnn:
        print("✓ Uses self.my_cnn for CNN model reference")
    
    # Verify torchvision.models is used for resnet152
    uses_torchvision = "torchvision.models" in content
    if uses_torchvision and has_resnet152:
        print("✓ Uses torchvision.models for resnet152 initialization")
    
    print("\n" + "=" * 70)
    print("Test Results:")
    
    all_checks_passed = (
        has_resnet152 and 
        has_correct_feature_size and 
        uses_myResnet and 
        has_myResnet_import and
        uses_my_cnn and
        uses_torchvision
    )
    
    if all_checks_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe dataloaderraw.py file correctly handles resnet152:")
        print("  - resnet152 is recognized as a valid CNN model")
        print("  - Feature size is set to 2048 (matching expected dimensions)")
        print("  - myResnet wrapper is used (not myDensenet)")
        print("  - torchvision.models is used for initialization")
        print("  - Code uses self.my_cnn for better generalization")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        if not has_resnet152:
            print("  - resnet152 handling not found")
        if not has_correct_feature_size:
            print("  - Feature size not set to 2048")
        if not uses_myResnet:
            print("  - myResnet wrapper not used")
        if not has_myResnet_import:
            print("  - myResnet import missing")
        if not uses_my_cnn:
            print("  - Not using self.my_cnn")
        if not uses_torchvision:
            print("  - torchvision.models not used")
        return False

def test_model_feature_sizes():
    """
    Verify that the documented feature sizes are correct in the code.
    """
    print("\n" + "=" * 70)
    print("Testing documented feature sizes...")
    print("=" * 70)
    
    with open('/home/runner/work/AoANet-Updated/AoANet-Updated/dataloaderraw.py', 'r') as f:
        content = f.read()
    
    expected_mappings = {
        'densenet121': 1024,
        'densenet161': 2208,
        'densenet169': 1664,
        'densenet201': 1920,
        'resnet101': 2048,
        'resnet152': 2048,
        'inception': 2048,
        'regnet': 3024,
    }
    
    all_correct = True
    for model, expected_size in expected_mappings.items():
        # Look for the pattern: elif cnn_model == 'model': ... self.feature_size = size
        import re
        pattern = rf"cnn_model == '{model}'.*?self\.feature_size = (\d+)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            actual_size = int(match.group(1))
            if actual_size == expected_size:
                print(f"✓ {model:15s}: feature_size = {expected_size}")
            else:
                print(f"✗ {model:15s}: expected {expected_size}, found {actual_size}")
                all_correct = False
        else:
            print(f"✗ {model:15s}: not found in code")
            all_correct = False
    
    print("\n" + "=" * 70)
    if all_correct:
        print("✓ All feature sizes are correctly configured")
    else:
        print("✗ Some feature sizes are incorrect")
    
    return all_correct

if __name__ == '__main__':
    result1 = test_dataloaderraw_code_structure()
    result2 = test_model_feature_sizes()
    
    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    if result1 and result2:
        print("✓ All tests passed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
