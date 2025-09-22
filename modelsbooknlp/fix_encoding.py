def fix_file_encoding(input_file, output_file):
    """Fix file encoding by trying common encodings and converting to UTF-8"""
    
    # Read the file in binary mode
    with open(input_file, 'rb') as f:
        raw_data = f.read()
    
    # Try common encodings
    text = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']:
        try:
            text = raw_data.decode(encoding)
            print(f"Successfully decoded with {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}")
            continue
    
    if text is None:
        # If all fail, decode with errors='replace'
        text = raw_data.decode('utf-8', errors='replace')
        print("Decoded with UTF-8 and replaced invalid characters")
    
    # Write as UTF-8
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"File saved as UTF-8: {output_file}")

if __name__ == "__main__":
    input_file = "Lord Of The Mysteries.txt"
    output_file = "Lord Of The Mysteries_utf8.txt"
    
    fix_file_encoding(input_file, output_file)
    print("Encoding conversion completed!")