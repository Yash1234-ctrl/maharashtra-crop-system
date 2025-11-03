"""
Simple script to validate that packages listed in requirements.txt are installed in the active Python
environment (or the project's .venv if you run it with that python). Non-invasive helper.
Usage (from project root using the venv python):
    .\.venv\Scripts\python.exe tools\validate_requirements.py
"""
import sys
from pathlib import Path

try:
    import pkg_resources
except Exception as e:
    print("pkg_resources not available. Install setuptools: python -m pip install setuptools")
    sys.exit(2)

project_root = Path(__file__).resolve().parents[1]
req_file = project_root / 'requirements.txt'
if not req_file.exists():
    print(f"requirements.txt not found at {req_file}")
    sys.exit(2)

with req_file.open('r', encoding='utf-8') as f:
    reqs = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
missing = []

for r in reqs:
    # handle simple requirement lines like 'package==x.y' or 'package>=x'
    try:
        req = pkg_resources.Requirement.parse(r)
    except Exception:
        # fallback: take the package name part
        name = r.split('==')[0].split('>=')[0].split('>')[0].split('<')[0].strip()
        req = None
        name_key = name.lower()
        if name_key not in installed:
            missing.append((name, None))
        continue

    name = req.name
    if name.lower() not in installed:
        missing.append((name, str(req.specifier)))

if not missing:
    print("All listed requirements appear to be installed in this Python environment.")
else:
    print("The following packages from requirements.txt are not installed in this environment:")
    for name, spec in missing:
        if spec:
            print(f"  - {name} {spec}")
        else:
            print(f"  - {name}")
    print('\nTip: run the project venv python, e.g.:')
    print(r"  .\.venv\Scripts\python.exe -m pip install -r requirements.txt")
    sys.exit(1)
