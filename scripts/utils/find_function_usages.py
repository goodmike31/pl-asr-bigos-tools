#!/usr/bin/env python3
"""
Utility script to find usages of a specific function across a codebase.
"""

import os
import re
import argparse
from pathlib import Path


def find_function_usages(root_dir, module_path, function_name):
    """
    Find all usages of a function in a codebase.
    
    Args:
        root_dir: Directory to search in
        module_path: Import path of the module containing the function
        function_name: Name of the function to search for
    
    Returns:
        List of files and line numbers where the function is used
    """
    results = []
    module_name = os.path.basename(module_path).replace(".py", "")
    
    for path in Path(root_dir).rglob("*.py"):
        if str(path) == module_path:
            continue  # Skip the file that defines the function
        
        try:
            with open(path, 'r') as file:
                content = file.read()
                lines = content.split('\n')
                
                # Look for direct imports like "from config_utils import get_config_run"
                direct_import_pattern = rf"from\s+.*{module_name}\s+import\s+.*{function_name}"
                # Look for module imports like "import config_utils" followed by "config_utils.get_config_run"
                module_import_pattern = rf"import\s+.*{module_name}"
                # Look for actual function calls
                function_call_pattern = rf"{function_name}\s*\("
                module_function_call_pattern = rf"{module_name}\.{function_name}\s*\("
                
                for i, line in enumerate(lines, 1):
                    if re.search(direct_import_pattern, line) or \
                       re.search(module_import_pattern, line) or \
                       re.search(function_call_pattern, line) or \
                       re.search(module_function_call_pattern, line):
                        results.append((str(path), i, line.strip()))
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Find usages of a function across the codebase.')
    parser.add_argument('--root', required=True, help='Root directory to search in')
    parser.add_argument('--module', required=True, help='Path to the module defining the function')
    parser.add_argument('--function', required=True, help='Name of the function to search for')
    
    args = parser.parse_args()
    
    usages = find_function_usages(args.root, args.module, args.function)
    print("Running...")
    if not usages:
        print(f"No usages of '{args.function}' found.")
        return
    
    print(f"Found {len(usages)} potential usages of '{args.function}':")
    for file_path, line_num, line_content in usages:
        print(f"{file_path}:{line_num}: {line_content}")


if __name__ == "__main__":
    main()
