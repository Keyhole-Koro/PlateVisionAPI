import os
import re
import pkg_resources

EXCLUDED_FOLDERS = {"freezed", "tests"}

def extract_imports_from_file(filepath):
    """Extract imported modules from a Python file."""
    imports = set()
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            match = re.match(r"^\s*(?:import|from)\s+([\w\.]+)", line)
            if match:
                imports.add(match.group(1).split(".")[0])  # Get the top-level module
    return imports

def traverse_files_and_collect_imports(root_folder, excluded_folders):
    """Traverse files and collect all imports."""
    all_imports = set()
    for root, dirs, files in os.walk(root_folder):
        print(f"Checking directory: {root}")
        # Skip excluded folders
        dirs[:] = [d for d in dirs if d not in excluded_folders]
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                all_imports.update(extract_imports_from_file(filepath))
    return all_imports

def filter_installed_packages(imports):
    """Filter imports to include only installed packages."""
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    return sorted(imports.intersection(installed_packages))

def generate_requirements_file(imports, output_file):
    """Generate a requirements.txt file."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("# Auto-generated requirements.txt\n")
        for package in imports:
            file.write(f"{package}\n")
    print(f"Requirements file generated at: {output_file}")

if __name__ == "__main__":
    # Prompt the user for the root path
    root_path = input("Enter the root path of the project (default is current directory): ").strip()
    if not root_path:
        root_path = os.getcwd()  # Use the current directory as the default

    if not os.path.isdir(root_path):
        print(f"Error: The specified path '{root_path}' is not a valid directory.")
        exit(1)

    output_file = os.path.join(root_path, "requirements.txt")

    print("Traversing files and collecting imports...")
    imports = traverse_files_and_collect_imports(root_path, EXCLUDED_FOLDERS)
    print(f"Found imports: {imports}")

    print("Filtering installed packages...")
    filtered_imports = filter_installed_packages(imports)
    print(f"Filtered imports: {filtered_imports}")

    print("Generating requirements.txt...")
    generate_requirements_file(filtered_imports, output_file)
