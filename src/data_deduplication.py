"""
Data Deduplication Module for Phoenix Protocol.
Implements pHash (Perceptual Hashing) with Hamming distance threshold
to identify and remove inter-split duplicates before training.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from tqdm import tqdm

try:
    from PIL import Image
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("Warning: imagehash not available. Install with: pip install imagehash")


class ImageDeduplicator:
    """
    Perceptual hash-based image deduplication to prevent data leakage.
    Uses pHash with Hamming distance threshold of 5 to identify near-duplicates.
    """
    
    def __init__(self, hamming_threshold: int = 5):
        """
        Initialize deduplicator.
        
        Args:
            hamming_threshold: Maximum Hamming distance for considering images as duplicates.
                              Default is 5 as specified in Phoenix Protocol.
        """
        if not IMAGEHASH_AVAILABLE:
            raise ImportError("imagehash library required. Install with: pip install imagehash")
        
        self.hamming_threshold = hamming_threshold
        self.image_hashes: Dict[str, 'imagehash.ImageHash'] = {}
        self.duplicate_groups: List[List[str]] = []
        
    def compute_phash(self, image_path: str) -> 'imagehash.ImageHash':
        """
        Compute perceptual hash for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Perceptual hash of the image
        """
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            return imagehash.phash(image, hash_size=8)
        except Exception as e:
            print(f"Error computing hash for {image_path}: {e}")
            return None
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict[str, 'imagehash.ImageHash']:
        """
        Scan directory and compute hashes for all images.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            Dictionary mapping image paths to their hashes
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if Path(f).suffix.lower() in image_extensions
            ]
        
        hashes = {}
        print(f"Computing perceptual hashes for {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Computing hashes"):
            phash = self.compute_phash(image_path)
            if phash is not None:
                hashes[image_path] = phash
        
        return hashes
    
    def find_duplicates(self, hash_dict: Dict[str, 'imagehash.ImageHash']) -> List[List[str]]:
        """
        Find duplicate images based on Hamming distance between hashes.
        
        Args:
            hash_dict: Dictionary mapping image paths to their hashes
            
        Returns:
            List of duplicate groups, where each group contains paths to similar images
        """
        print(f"Finding duplicates with Hamming threshold: {self.hamming_threshold}...")
        
        image_paths = list(hash_dict.keys())
        n = len(image_paths)
        visited = set()
        duplicate_groups = []
        
        for i in tqdm(range(n), desc="Finding duplicates"):
            if i in visited:
                continue
                
            group = [image_paths[i]]
            hash_i = hash_dict[image_paths[i]]
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                    
                hash_j = hash_dict[image_paths[j]]
                hamming_dist = hash_i - hash_j  # imagehash library overloads - operator
                
                if hamming_dist <= self.hamming_threshold:
                    group.append(image_paths[j])
                    visited.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
            
            visited.add(i)
        
        return duplicate_groups
    
    def detect_cross_split_duplicates(
        self, 
        train_dir: str, 
        val_dir: str, 
        test_dir: str
    ) -> Dict:
        """
        Detect duplicates across train/val/test splits.
        Critical for preventing data leakage.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            test_dir: Test data directory
            
        Returns:
            Dictionary with cross-split duplicate information
        """
        print("\n" + "="*80)
        print("PHOENIX PROTOCOL: CROSS-SPLIT DUPLICATE DETECTION")
        print("="*80 + "\n")
        
        # Scan all directories
        print("Scanning training set...")
        train_hashes = self.scan_directory(train_dir) if os.path.exists(train_dir) else {}
        
        print("\nScanning validation set...")
        val_hashes = self.scan_directory(val_dir) if os.path.exists(val_dir) else {}
        
        print("\nScanning test set...")
        test_hashes = self.scan_directory(test_dir) if os.path.exists(test_dir) else {}
        
        # Combine all hashes with split labels
        all_hashes = {}
        split_labels = {}
        
        for path, hash_val in train_hashes.items():
            all_hashes[path] = hash_val
            split_labels[path] = 'train'
        
        for path, hash_val in val_hashes.items():
            all_hashes[path] = hash_val
            split_labels[path] = 'val'
        
        for path, hash_val in test_hashes.items():
            all_hashes[path] = hash_val
            split_labels[path] = 'test'
        
        # Find all duplicates
        all_duplicate_groups = self.find_duplicates(all_hashes)
        
        # Filter for cross-split duplicates only
        cross_split_duplicates = []
        for group in all_duplicate_groups:
            splits_in_group = set(split_labels[path] for path in group)
            if len(splits_in_group) > 1:
                cross_split_duplicates.append({
                    'images': group,
                    'splits': [split_labels[path] for path in group],
                    'count': len(group)
                })
        
        # Generate statistics
        stats = {
            'total_images': len(all_hashes),
            'train_images': len(train_hashes),
            'val_images': len(val_hashes),
            'test_images': len(test_hashes),
            'total_duplicate_groups': len(all_duplicate_groups),
            'cross_split_duplicate_groups': len(cross_split_duplicates),
            'hamming_threshold': self.hamming_threshold
        }
        
        # Print summary
        print("\n" + "="*80)
        print("DEDUPLICATION RESULTS")
        print("="*80)
        print(f"Total images scanned: {stats['total_images']}")
        print(f"  - Training: {stats['train_images']}")
        print(f"  - Validation: {stats['val_images']}")
        print(f"  - Test: {stats['test_images']}")
        print(f"\nTotal duplicate groups found: {stats['total_duplicate_groups']}")
        print(f"Cross-split duplicates (DATA LEAKAGE): {stats['cross_split_duplicate_groups']}")
        
        if cross_split_duplicates:
            print(f"\n⚠️  WARNING: {stats['cross_split_duplicate_groups']} cross-split duplicate groups detected!")
            print("This indicates potential data leakage between train/val/test sets.")
        else:
            print("\n✓ No cross-split duplicates detected. Dataset is clean.")
        
        print("="*80 + "\n")
        
        return {
            'cross_split_duplicates': cross_split_duplicates,
            'all_duplicate_groups': all_duplicate_groups,
            'statistics': stats
        }
    
    def save_deduplication_report(self, results: Dict, output_path: str):
        """
        Save deduplication results to JSON file.
        
        Args:
            results: Results from detect_cross_split_duplicates
            output_path: Path to save the report
        """
        # Convert for JSON serialization
        serializable_results = {
            'statistics': results['statistics'],
            'cross_split_duplicate_count': len(results['cross_split_duplicates']),
            'cross_split_duplicates': results['cross_split_duplicates']
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Deduplication report saved to: {output_path}")
    
    def remove_duplicates_strategy(
        self, 
        duplicate_groups: List[Dict], 
        priority_order: List[str] = ['train', 'val', 'test']
    ) -> Set[str]:
        """
        Determine which images to keep based on split priority.
        For cross-split duplicates, keep the image in the highest priority split.
        
        Args:
            duplicate_groups: List of duplicate groups from detect_cross_split_duplicates
            priority_order: Order of splits by priority (default: keep in train, remove from val/test)
            
        Returns:
            Set of image paths that should be removed
        """
        images_to_remove = set()
        
        for group_info in duplicate_groups:
            images = group_info['images']
            splits = group_info['splits']
            
            # Find the image with highest priority split
            keep_idx = None
            keep_priority = len(priority_order)
            
            for idx, (img, split) in enumerate(zip(images, splits)):
                priority = priority_order.index(split) if split in priority_order else len(priority_order)
                if priority < keep_priority:
                    keep_priority = priority
                    keep_idx = idx
            
            # Mark all others for removal
            for idx, img in enumerate(images):
                if idx != keep_idx:
                    images_to_remove.add(img)
        
        return images_to_remove


def deduplicate_dataset(
    data_dir: str,
    hamming_threshold: int = 5,
    output_report: str = None,
    dry_run: bool = True
) -> Dict:
    """
    Main function to deduplicate dataset across train/val/test splits.
    
    Args:
        data_dir: Base data directory containing train/val/test subdirectories
        hamming_threshold: Hamming distance threshold for considering duplicates
        output_report: Path to save deduplication report (optional)
        dry_run: If True, only report duplicates without removing files
        
    Returns:
        Dictionary with deduplication results
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            print(f"Warning: Directory not found: {d}")
    
    # Create deduplicator
    deduplicator = ImageDeduplicator(hamming_threshold=hamming_threshold)
    
    # Detect cross-split duplicates
    results = deduplicator.detect_cross_split_duplicates(train_dir, val_dir, test_dir)
    
    # Save report if requested
    if output_report:
        deduplicator.save_deduplication_report(results, output_report)
    
    # Optionally remove duplicates
    if not dry_run and results['cross_split_duplicates']:
        images_to_remove = deduplicator.remove_duplicates_strategy(
            results['cross_split_duplicates']
        )
        
        print(f"\nRemoving {len(images_to_remove)} duplicate images...")
        for img_path in tqdm(images_to_remove, desc="Removing duplicates"):
            try:
                os.remove(img_path)
                print(f"Removed: {img_path}")
            except Exception as e:
                print(f"Error removing {img_path}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    parser = argparse.ArgumentParser(
        description='Phoenix Protocol: Image Deduplication for Medical Imaging Datasets'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=config.DATA_DIR,
        help='Base data directory containing train/val/test splits'
    )
    parser.add_argument(
        '--hamming-threshold',
        type=int,
        default=5,
        help='Hamming distance threshold (default: 5)'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default=os.path.join(config.RESULTS_DIR, 'deduplication_report.json'),
        help='Path to save deduplication report'
    )
    parser.add_argument(
        '--remove-duplicates',
        action='store_true',
        help='Actually remove duplicate files (default: dry run)'
    )
    
    args = parser.parse_args()
    
    # Run deduplication
    results = deduplicate_dataset(
        data_dir=args.data_dir,
        hamming_threshold=args.hamming_threshold,
        output_report=args.output_report,
        dry_run=not args.remove_duplicates
    )
    
    print("\n" + "="*80)
    print("PHOENIX PROTOCOL DEDUPLICATION COMPLETE")
    print("="*80)
