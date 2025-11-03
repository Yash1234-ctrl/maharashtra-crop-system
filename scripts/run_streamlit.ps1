<#
PowerShell helper to start the Streamlit app using the project's virtual environment.
Non-invasive: does not modify project code. Uses .venv\Scripts\python.exe to run Streamlit.
Usage:
    Set-Location -Path 'E:\GC Update'
    .\scripts\run_streamlit.ps1 -Port 8501
#>
param(
    [string]$Workspace = (Get-Location).Path,
    [int]$Port = 8501
)

$pythonExe = Join-Path $Workspace ".venv\Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Error "Virtualenv python not found at $pythonExe. Run scripts\setup_env.ps1 first."
    exit 1
}

# Make sure pip is recent and streamlit is installed (installing only streamlit is fast compared to full requirements)
Write-Host "Ensuring pip and streamlit are available in the venv (this will install streamlit if missing)..."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install streamlit --upgrade

$script = Join-Path $Workspace "maharashtra_crop_system.py"
if (-Not (Test-Path $script)) {
    Write-Error "Streamlit app not found at $script"
    exit 1
}

Write-Host "Starting Streamlit app (this will attach to the current terminal). Press Ctrl+C to stop."
& $pythonExe -m streamlit run $script --server.port $Port
