import os
import subprocess

# Configuratie
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"base_path: {base_path}")
utils_path = os.path.join(base_path, "utils")
venv_python = os.path.join(os.path.dirname(base_path), ".venv", "Scripts", "python.exe")

bestanden = [
    os.path.join(base_path, "data", "testbatch1", "prepare_stock_and_daily_data.py"),
    os.path.join(utils_path, "generate_relation_new.py"),
    os.path.join(utils_path, "normalize_data.py"),
    os.path.join(utils_path, "generate_data.py")
]

for bestand in bestanden:
    print(f"Running {bestand}...")
    result = subprocess.run([venv_python, bestand], capture_output=True, text=True)

    # Print output naar console (optioneel)
    print(f"--- Output van {bestand} ---")
    print(result.stdout)

    # Als er een fout was
    if result.stderr:
        print(f"--- Fout in {bestand} ---")
        print(result.stderr)
