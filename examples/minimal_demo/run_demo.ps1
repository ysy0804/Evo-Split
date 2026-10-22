$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Build = Join-Path $Root "build-demo"
$Results = Join-Path $Root "demo-results"
New-Item -ItemType Directory -Force -Path $Results | Out-Null
cmake -S $Root -B $Build -DEVOSPLIT_WITH_MNN=OFF -DEVOSPLIT_BUILD_TESTS=ON -DEVOSPLIT_BUILD_DEMO=ON
cmake --build $Build --config Release
$Exe = Join-Path $Build "Release\evosplit_demo.exe"
if (-not (Test-Path $Exe)) { $Exe = Join-Path $Build "evosplit_demo.exe" }
& $Exe --output (Join-Path $Results "demo_trace.csv")
python (Join-Path $PSScriptRoot "plot_demo.py") --input (Join-Path $Results "demo_trace.csv") --output (Join-Path $Results "demo_summary.png")

