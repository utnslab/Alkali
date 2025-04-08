#!/usr/bin/env python3
import re
import json
import sys
import argparse

def parse_c_file(file_content):
    """
    Parse the provided C file content to extract struct definitions
    that contain BITS_FIELD macros. For each such struct, extract the
    integer values from the first parameter of BITS_FIELD.
    
    Args:
        file_content (str): The content of the C file.
    
    Returns:
        dict: A dictionary where keys are struct names and values are
              lists of integer values extracted from BITS_FIELD.
    """
    # Regex to capture struct definitions: struct <name> { ... };
    struct_pattern = re.compile(r'struct\s+(\w+)\s*\{(.*?)\};', re.DOTALL)
    # Regex to capture BITS_FIELD(integer, field_name)
    bits_field_pattern = re.compile(r'BITS_FIELD\s*\(\s*(\d+)\s*,\s*\w+\s*\)')

    result = {}
    for struct_match in struct_pattern.finditer(file_content):
        struct_name = struct_match.group(1)
        struct_body = struct_match.group(2)
        
        # Find all occurrences of BITS_FIELD in the struct body
        numbers = [int(num) for num in bits_field_pattern.findall(struct_body)]
        if numbers:
            result[struct_name] = numbers
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Convert C file struct BITS_FIELD definitions to a JSON dictionary."
    )
    parser.add_argument("input_file", help="Path to the input C file")
    parser.add_argument(
        "-o", "--output", help="Path to the output JSON file (default: stdout)", default=None
    )
    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            file_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    parsed_data = parse_c_file(file_content)
    json_output = json.dumps(parsed_data, indent=4)
    
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(json_output)
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
    else:
        print(json_output)

if __name__ == "__main__":
    main()

