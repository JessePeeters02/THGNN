import os
import subprocess
import sys

# === Configuratie ===

# Bepaal het basispad van je project (één map omhoog vanaf dit script)
base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
print(f"Base path: {base_path}")
utils_path = os.path.join(base_path, "THGNN", "utils")
print(f"Utils path: {utils_path}")

# Gebruik gewoon 'python' als executable (jij gebruikt Python 3.9 op Windows)
python_executable = "python"

# Voeg hier je scripts toe, in de juiste volgorde
scripts_to_run = [
    os.path.join(utils_path, "generate_relations_and_create_snapshots_onlycosine.py"),
    os.path.join(utils_path, "generate_relations_DynamiSE_cosineplusDSC.py"),
    os.path.join(utils_path, "generate_relations_DynamiSE_SSADSC.py")
]

# === Pre-check: bestaan alle scripts? ===
print("\n🔎 Controleren of alle scripts bestaan...")
all_exist = True
for script in scripts_to_run:
    if not os.path.exists(script):
        print(f"❌ Script bestaat niet: {script}")
        all_exist = False
if not all_exist:
    print("⛔️ Afgebroken: één of meerdere scripts ontbreken.")
    sys.exit(1)  # Exit met error code

# === Scripts uitvoeren ===
for script in scripts_to_run:
    print(f"\n🔄 Running script: {script}...\n")
    result = subprocess.run([python_executable, script], capture_output=True, text=True)

    # Toon output
    print(f"✅ Output van {os.path.basename(script)}:\n{result.stdout}")

    # Toon foutmeldingen
    if result.stderr:
        print(f"❌ Foutmelding in {os.path.basename(script)}:\n{result.stderr}")
        print("⛔️ Script beëindigd wegens fout.")
        break
