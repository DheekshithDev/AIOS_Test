import base64

def save_file_from_base64(encoded_str: str, filename: str):
    """
    Decode a base64 string and save it as a file.
    
    Args:
        encoded_str (str): The base64 string (your DB content).
        filename (str): The output file name, e.g. "config.json", "script.py", "notes.txt".
    """
    # Decode
    decoded_bytes = base64.b64decode(encoded_str)
    
    # Write to file
    with open(filename, "wb") as f:
        f.write(decoded_bytes)
    print(f"Saved: {filename}")

# Example usage
b64_str = """YXV0b2dlbi1hZ2VudGNoYXQ9PTAuMi40MAo="""

# Save as JSON
save_file_from_base64(b64_str, "meta_requirements.txt")
