"""
Script to aggregate all Python files from data/raw/code into a single training corpus.
This ensures the model trains on actual code rather than mixed/tiny datasets.
"""

import os
import glob
from pathlib import Path

def create_code_corpus():
    """
    Aggregates all .py files from data/raw/code into data/processed/code_corpus.txt
    """
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Paths
    code_dir = project_root / "data" / "raw" / "code"
    output_dir = project_root / "data" / "processed"
    output_file = output_dir / "code_corpus.txt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all Python files
    py_files = sorted(glob.glob(str(code_dir / "*.py")))
    
    if not py_files:
        print("WARNING: No Python files found!")
        return
    
    print(f"Found {len(py_files)} Python files")
    
    # Aggregate content
    files_processed = 0
    files_skipped = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as in_f:
                    content = in_f.read()
                
                # Skip empty files or files with just "404: Not Found"
                if not content.strip() or content.strip() == "404: Not Found" or len(content) < 10:
                    print(f"Skipping (empty/invalid): {os.path.basename(py_file)}")
                    files_skipped += 1
                    continue
                
                print(f"Processing: {os.path.basename(py_file)} ({len(content)} chars)")
                
                # Write file header for context
                out_f.write(f"# File: {os.path.basename(py_file)}\n")
                out_f.write(content)
                out_f.write("\n\n")  # Separate files with blank lines
                files_processed += 1
                    
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
                files_skipped += 1
    
    # Report statistics
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = len(content.split('\n'))
        chars = len(content)
        words = len(content.split())
    
    print(f"\n✓ Code corpus created: {output_file}")
    print(f"  - Files processed: {files_processed}")
    print(f"  - Files skipped: {files_skipped} (empty/invalid)")
    print(f"  - Lines: {lines:,}")
    print(f"  - Characters: {chars:,}")
    print(f"  - Words: {words:,}")
    
    return output_file

if __name__ == "__main__":
    create_code_corpus()

