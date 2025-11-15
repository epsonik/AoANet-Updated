"""
Demo script showing how the metric calculation feature works.
This creates a sample CSV with mock data to demonstrate the output format.
"""
import os
import sys
import csv

# Add coco-caption to path
sys.path.append("coco-caption")


def demo_metrics_csv():
    """Create a sample evaluation_metrics.csv with demo data."""
    print("=" * 60)
    print("Demo: Metric Calculation Feature")
    print("=" * 60)
    print()
    
    # Create demo output directory
    output_dir = '/tmp/demo_metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, 'evaluation_metrics.csv')
    
    # Sample data (simulating real metric outputs)
    sample_data = [
        {
            'image_id': 391895,
            'predicted_caption': 'a bicycle with a clock as the front wheel',
            'BLEU_1': 0.8571,
            'BLEU_2': 0.7143,
            'BLEU_3': 0.5714,
            'BLEU_4': 0.4286,
            'METEOR': 0.3456,
            'CIDEr': 1.2345,
            'ROUGE_L': 0.6789
        },
        {
            'image_id': 203564,
            'predicted_caption': 'a dog sitting on a couch',
            'BLEU_1': 0.9000,
            'BLEU_2': 0.7500,
            'BLEU_3': 0.6000,
            'BLEU_4': 0.4500,
            'METEOR': 0.4123,
            'CIDEr': 1.5678,
            'ROUGE_L': 0.7234
        },
        {
            'image_id': 123456,
            'predicted_caption': 'a cat laying on top of a bed',
            'BLEU_1': 0.8750,
            'BLEU_2': 0.7000,
            'BLEU_3': 0.5500,
            'BLEU_4': 0.4000,
            'METEOR': 0.3890,
            'CIDEr': 1.4567,
            'ROUGE_L': 0.7000
        }
    ]
    
    # Write to CSV
    fieldnames = ['image_id', 'predicted_caption', 'BLEU_1', 'BLEU_2', 
                 'BLEU_3', 'BLEU_4', 'METEOR', 'CIDEr', 'ROUGE_L']
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)
    
    print(f"Created sample CSV file: {csv_file}")
    print()
    print("CSV Content:")
    print("-" * 60)
    
    # Display the CSV content
    with open(csv_file, 'r') as f:
        print(f.read())
    
    print("-" * 60)
    print()
    print("Explanation of Metrics:")
    print()
    print("  BLEU (1-4): Measures n-gram overlap between predicted and reference")
    print("              captions. Higher is better (range 0-1).")
    print()
    print("  METEOR:     Considers synonyms and stemming. Higher is better")
    print("              (range 0-1).")
    print()
    print("  CIDEr:      Consensus-based metric weighted by TF-IDF. Higher is")
    print("              better (typically 0-10).")
    print()
    print("  ROUGE_L:    Based on longest common subsequence. Higher is better")
    print("              (range 0-1).")
    print()
    print("=" * 60)
    print(f"Demo complete! Sample CSV saved to: {csv_file}")
    print("=" * 60)
    
    return csv_file


def demo_image_id_extraction():
    """Demonstrate image ID extraction from filenames."""
    print()
    print("=" * 60)
    print("Demo: Image ID Extraction")
    print("=" * 60)
    print()
    
    # Import the function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # We can't import from visualize.py without torch, so we'll define it here
    def extract_image_id_from_filename(filename):
        """Extract COCO image ID from filename."""
        basename = os.path.basename(filename)
        if 'COCO' in basename:
            parts = basename.split('_')
            if len(parts) >= 3:
                id_str = parts[-1].split('.')[0]
                try:
                    return int(id_str)
                except ValueError:
                    pass
        try:
            return int(basename.split('.')[0])
        except ValueError:
            pass
        return None
    
    test_filenames = [
        'COCO_val2014_000000391895.jpg',
        'COCO_val2014_000000203564.jpg',
        '/path/to/COCO_val2014_000000123456.jpg',
        '999999.png',
    ]
    
    print("Extracting image IDs from filenames:")
    print()
    
    for filename in test_filenames:
        image_id = extract_image_id_from_filename(filename)
        print(f"  {filename}")
        print(f"    → Image ID: {image_id}")
        print()
    
    print("=" * 60)


if __name__ == '__main__':
    print()
    demo_metrics_csv()
    demo_image_id_extraction()
    print()
