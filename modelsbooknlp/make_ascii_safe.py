def make_ascii_safe(input_file, output_file):
    """Convert text file to ASCII-safe version by removing/replacing problematic characters"""
    
    # Read the UTF-8 file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original file size: {len(text)} characters")
    
    # Replace problematic characters
    # Common problematic characters and their replacements
    replacements = {
        '"': '"',  # Left double quotation mark
        '"': '"',  # Right double quotation mark
        ''': "'",  # Left single quotation mark
        ''': "'",  # Right single quotation mark
        '–': '-',  # En dash
        '—': '--', # Em dash
        '…': '...', # Horizontal ellipsis
        '•': '*',  # Bullet point
        '°': ' degrees',  # Degree symbol
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode('ascii')
    
    print(f"Cleaned file size: {len(text)} characters")
    
    # Write as ASCII
    with open(output_file, 'w', encoding='ascii') as f:
        f.write(text)
    
    print(f"ASCII-safe file saved: {output_file}")

if __name__ == "__main__":
    input_file = "Lord Of The Mysteries_utf8.txt"
    output_file = "Lord Of The Mysteries_clean.txt"
    
    make_ascii_safe(input_file, output_file)
    print("ASCII conversion completed!")