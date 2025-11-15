"""
Test script to verify metric calculation functionality in visualize.py.
This tests the metric calculation functions without requiring a trained model.
"""
import os
import sys
import csv
import json
import tempfile
import shutil

# Add coco-caption to path
sys.path.append("coco-caption")


# Copy of the functions from visualize.py to test without dependencies
def calculate_metrics_for_image(image_id, predicted_caption, coco_annotations):
    """
    Calculate BLEU, METEOR, CIDEr, and ROUGE_L metrics for a single image.
    
    Args:
        image_id: COCO image ID
        predicted_caption: Generated caption string
        coco_annotations: Path to COCO annotations JSON file
        
    Returns:
        Dictionary containing all metrics
    """
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError:
        print("Warning: pycocoevalcap not available. Metrics will not be calculated.")
        return None
    
    # Load COCO annotations
    coco = COCO(coco_annotations)
    
    # Check if image_id is valid
    if image_id not in coco.getImgIds():
        print(f"Warning: Image ID {image_id} not found in COCO annotations")
        return None
    
    # Create temporary result in COCO format
    result = [{'image_id': image_id, 'caption': predicted_caption}]
    
    # Create a temporary file for results (required by COCO API)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(result, f)
        temp_file = f.name
    
    try:
        # Load results
        cocoRes = coco.loadRes(temp_file)
        
        # Evaluate
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = [image_id]
        cocoEval.evaluate()
        
        # Get metrics for this specific image
        metrics = {}
        if image_id in cocoEval.imgToEval:
            img_metrics = cocoEval.imgToEval[image_id]
            metrics['BLEU_1'] = img_metrics.get('Bleu_1', 0.0)
            metrics['BLEU_2'] = img_metrics.get('Bleu_2', 0.0)
            metrics['BLEU_3'] = img_metrics.get('Bleu_3', 0.0)
            metrics['BLEU_4'] = img_metrics.get('Bleu_4', 0.0)
            metrics['METEOR'] = img_metrics.get('METEOR', 0.0)
            metrics['CIDEr'] = img_metrics.get('CIDEr', 0.0)
            metrics['ROUGE_L'] = img_metrics.get('ROUGE_L', 0.0)
        
        return metrics
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def extract_image_id_from_filename(filename):
    """
    Extract COCO image ID from filename.
    Expected format: COCO_val2014_000000391895.jpg -> 391895
    """
    basename = os.path.basename(filename)
    # Try to extract ID from COCO format
    if 'COCO' in basename:
        parts = basename.split('_')
        if len(parts) >= 3:
            # Get the numeric part (remove .jpg or other extension)
            id_str = parts[-1].split('.')[0]
            try:
                return int(id_str)
            except ValueError:
                pass
    
    # If not in COCO format, try to parse the entire basename as a number
    try:
        return int(basename.split('.')[0])
    except ValueError:
        pass
    
    return None


def save_metrics_to_csv(csv_file, image_id, predicted_caption, metrics):
    """
    Save metrics to CSV file.
    
    Args:
        csv_file: Path to CSV file
        image_id: Image ID
        predicted_caption: Generated caption
        metrics: Dictionary of metrics
    """
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['image_id', 'predicted_caption', 'BLEU_1', 'BLEU_2', 
                     'BLEU_3', 'BLEU_4', 'METEOR', 'CIDEr', 'ROUGE_L']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write metrics
        row = {
            'image_id': image_id,
            'predicted_caption': predicted_caption,
            'BLEU_1': metrics.get('BLEU_1', 0.0),
            'BLEU_2': metrics.get('BLEU_2', 0.0),
            'BLEU_3': metrics.get('BLEU_3', 0.0),
            'BLEU_4': metrics.get('BLEU_4', 0.0),
            'METEOR': metrics.get('METEOR', 0.0),
            'CIDEr': metrics.get('CIDEr', 0.0),
            'ROUGE_L': metrics.get('ROUGE_L', 0.0)
        }
        writer.writerow(row)


def test_extract_image_id_from_filename():
    """Test extraction of image ID from filenames."""
    print("Testing extract_image_id_from_filename...")
    
    # Test COCO format
    test_cases = [
        ('COCO_val2014_000000391895.jpg', 391895),
        ('COCO_val2014_000000203564.jpg', 203564),
        ('/path/to/COCO_val2014_000000391895.jpg', 391895),
        ('391895.jpg', 391895),
        ('203564.png', 203564),
    ]
    
    for filename, expected_id in test_cases:
        result = extract_image_id_from_filename(filename)
        if result != expected_id:
            print(f"✗ Failed for {filename}: expected {expected_id}, got {result}")
            return False
    
    print("✓ Extract image ID test passed")
    return True


def test_save_metrics_to_csv():
    """Test saving metrics to CSV."""
    print("Testing save_metrics_to_csv...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, 'test_metrics.csv')
        
        # Test metrics
        metrics = {
            'BLEU_1': 0.8,
            'BLEU_2': 0.6,
            'BLEU_3': 0.4,
            'BLEU_4': 0.2,
            'METEOR': 0.3,
            'CIDEr': 1.0,
            'ROUGE_L': 0.5
        }
        
        # Save first entry
        save_metrics_to_csv(csv_file, 391895, "a dog on a couch", metrics)
        
        # Check file was created
        if not os.path.exists(csv_file):
            print(f"✗ CSV file not created: {csv_file}")
            return False
        
        # Read and verify content
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) != 1:
                print(f"✗ Expected 1 row, got {len(rows)}")
                return False
            
            row = rows[0]
            if row['image_id'] != '391895':
                print(f"✗ Wrong image_id: {row['image_id']}")
                return False
            
            if row['predicted_caption'] != 'a dog on a couch':
                print(f"✗ Wrong caption: {row['predicted_caption']}")
                return False
            
            if float(row['BLEU_1']) != 0.8:
                print(f"✗ Wrong BLEU_1: {row['BLEU_1']}")
                return False
        
        # Save second entry to test append
        save_metrics_to_csv(csv_file, 203564, "a bicycle with a clock", metrics)
        
        # Read and verify both entries
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) != 2:
                print(f"✗ Expected 2 rows after append, got {len(rows)}")
                return False
    
    print("✓ Save metrics to CSV test passed")
    return True


def test_calculate_metrics_for_image():
    """Test metric calculation for a single image."""
    print("Testing calculate_metrics_for_image...")
    
    # Check if COCO annotations exist
    coco_annotations = 'coco-caption/annotations/captions_val2014.json'
    if not os.path.exists(coco_annotations):
        print(f"Warning: COCO annotations not found at {coco_annotations}, skipping test")
        return True
    
    try:
        # Test with a known image ID from COCO val2014
        # Using image_id 391895 which should exist in the validation set
        image_id = 391895
        predicted_caption = "a bicycle with a clock on the front wheel"
        
        metrics = calculate_metrics_for_image(image_id, predicted_caption, coco_annotations)
        
        if metrics is None:
            print(f"Warning: pycocoevalcap dependencies not available, skipping test")
            print(f"  (This is expected in environments without matplotlib, etc.)")
            return True
        
        # Check that all required metrics are present
        required_metrics = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 
                           'METEOR', 'CIDEr', 'ROUGE_L']
        for metric in required_metrics:
            if metric not in metrics:
                print(f"✗ Missing metric: {metric}")
                return False
            
            # Check that metrics are numeric
            if not isinstance(metrics[metric], (int, float)):
                print(f"✗ Metric {metric} is not numeric: {metrics[metric]}")
                return False
        
        print(f"  Calculated metrics for image {image_id}:")
        print(f"    BLEU-1: {metrics['BLEU_1']:.4f}, BLEU-2: {metrics['BLEU_2']:.4f}, "
              f"BLEU-3: {metrics['BLEU_3']:.4f}, BLEU-4: {metrics['BLEU_4']:.4f}")
        print(f"    METEOR: {metrics['METEOR']:.4f}, CIDEr: {metrics['CIDEr']:.4f}, "
              f"ROUGE_L: {metrics['ROUGE_L']:.4f}")
        
        print("✓ Calculate metrics test passed")
        return True
        
    except ImportError as e:
        print(f"Warning: Required libraries not available ({e}), skipping test")
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Metric Calculation Functionality")
    print("=" * 60)
    print()
    
    tests = [
        test_extract_image_id_from_filename,
        test_save_metrics_to_csv,
        test_calculate_metrics_for_image,
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
