param(
    [ValidateSet('pilot', 'full')][string]$Stage = 'pilot',
    [string]$Python = '',
    [string]$OutputRoot = '',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_sacm_experiment.py'
if (-not $Python) { $Python = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe' }
$conditions = if ($Stage -eq 'pilot') { @('pbm', 'zero', 'heterogeneous') } else { @('pbm', 'zero', 'heterogeneous', 'global', 'low', 'high', 'weak_cue') }
$seeds = if ($Stage -eq 'pilot') { @(2023) } else { @(2023, 2024, 2025, 2026, 2027) }
foreach ($seed in $seeds) {
    foreach ($condition in $conditions) {
        $runArgs = @($runner, '--stage', $Stage, '--condition', $condition, '--seed', "$seed", '--python', $Python)
        if ($OutputRoot) { $runArgs += @('--output-root', $OutputRoot) }
        if ($Execute) { $runArgs += '--execute' }
        & $Python @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Stopped after failure: $Stage / $condition / $seed (exit $LASTEXITCODE)" }
    }
}
