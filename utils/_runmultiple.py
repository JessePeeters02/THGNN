import os
import subprocess
import sys
from datetime import datetime

# === Configuratie ===

# Bepaal het basispad van je project
base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
utils_path = os.path.join(base_path, "THGNN", "utils")

# Pad naar logbestand
log_file_path = os.path.join(base_path, "night_run_log.txt")
log_file = open(log_file_path, "w", encoding="utf-8")

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()

log(f"📁 Base path: {base_path}")
log(f"📁 Utils path: {utils_path}")

# Gebruik gewoon 'python' als executable
python_executable = "python"

# Voeg hier je scripts toe, in de juiste volgorde
scripts_to_run = [
    # os.path.join(utils_path, "generate_relations_and_create_snapshots_onlycosine.py"),
    os.path.join(utils_path, "generate_relations_DynamiSE_cosineplusDSC.py"),
    os.path.join(utils_path, "generate_relations_DynamiSE_SSADSC.py")
]

# === Pre-check: bestaan alle scripts? ===
log("\n🔎 Controleren of alle scripts bestaan...")
all_exist = True
for script in scripts_to_run:
    if not os.path.exists(script):
        log(f"❌ Script bestaat niet: {script}")
        all_exist = False
if not all_exist:
    log("⛔️ Afgebroken: één of meerdere scripts ontbreken.")
    log_file.close()
    sys.exit(1)

# === Scripts uitvoeren ===
for script in scripts_to_run:
    log(f"\n🔄 Starten van script: {os.path.basename(script)}")

    start_time = datetime.now()
    result = subprocess.run([python_executable, script], capture_output=True, text=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Output loggen
    if result.stdout.strip():
        log(f"✅ STDOUT:\n{result.stdout.strip()}")
    else:
        log("ℹ️ Geen standaarduitvoer.")

    # Error loggen
    if result.stderr.strip():
        log(f"❌ STDERR:\n{result.stderr.strip()}")
        log("⛔️ Script werd afgebroken wegens fout.")
        break
    else:
        log("✅ Script succesvol afgerond zonder fouten.")

    log(f"⏱️ Duur: {duration:.1f} seconden")

log("✅ Alle taken beëindigd.")
log_file.close()
