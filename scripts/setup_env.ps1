<#
PowerShell helper to create/upgrade virtual environment and install requirements.
Non-invasive: does not modify project code. Place in `scripts/` and run from project root.
Usage (PowerShell):
    Set-Location -Path 'E:\GC Update'
    .\scripts\setup_env.ps1
#>
param(
    [string]$Workspace = (Get-Location).Path
)

$venvPath = Join-Path $Workspace ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-Not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment at: $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at: $venvPath"
}

# Use the venv's python to upgrade pip and install requirements
if (Test-Path $pythonExe) {
    & $pythonExe -m pip install --upgrade pip
    $reqFile = Join-Path $Workspace "requirements.txt"
    if (Test-Path $reqFile) {
        Write-Host "Installing packages from requirements.txt (may take several minutes)..."
        & $pythonExe -m pip install -r $reqFile
        Write-Host "Installation attempt finished. If large packages (tensorflow/streamlit) stalled, re-run this script and allow it to complete."
    } else {
        Write-Error "requirements.txt not found at $reqFile"
    }
    Write-Host "To activate the virtual environment in this session run:`n    . $venvPath\Scripts\Activate.ps1"
} else {
    Write-Error "Virtualenv creation failed or python not found in $venvPath. Ensure Python is on PATH and try again."
}
