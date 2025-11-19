"""
Script to create a mixed corpus with both code and literature.
This provides better language understanding while maintaining code focus.
"""

import os
from pathlib import Path

def create_mixed_corpus(code_weight=3):
    """
    Creates a mixed corpus with code repeated multiple times to balance against books.
    
    Args:
        code_weight: How many times to include code corpus vs books (3 = 3:1 ratio)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Input paths
    code_corpus = project_root / "data" / "processed" / "code_corpus.txt"
    books_dir = project_root / "data" / "raw" / "books"
    
    # Output path
    output_file = project_root / "data" / "processed" / "mixed_corpus.txt"
    
    if not code_corpus.exists():
        print("ERROR: code_corpus.txt not found. Run create_code_corpus.py first!")
        return
    
    print(f"Creating mixed corpus with {code_weight}x code weight...")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Add code multiple times (weighted)
        with open(code_corpus, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        for i in range(code_weight):
            out_f.write(f"# === CODE SECTION {i+1}/{code_weight} ===\n\n")
            out_f.write(code_content)
            out_f.write("\n\n")
        
        # Add books (sample from each to keep size manageable)
        book_files = sorted(books_dir.glob("*.txt"))
        
        for book_file in book_files:
            print(f"Adding excerpt from: {book_file.name}")
            
            with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Take first 1000 lines from each book for variety
                excerpt = ''.join(lines[:1000])
            
            out_f.write(f"\n# === BOOK EXCERPT: {book_file.name} ===\n\n")
            out_f.write(excerpt)
            out_f.write("\n\n")
    
    # Report statistics
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = len(content.split('\n'))
        chars = len(content)
        words = len(content.split())
    
    print(f"\n✓ Mixed corpus created: {output_file}")
    print(f"  - Lines: {lines:,}")
    print(f"  - Characters: {chars:,}")
    print(f"  - Words: {words:,}")
    
    return output_file

if __name__ == "__main__":
    create_mixed_corpus(code_weight=3)

