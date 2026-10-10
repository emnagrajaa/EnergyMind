# EnergyMind - task runner for Windows (no `make` required).
# Mirrors the Makefile targets. Usage:  .\tasks.ps1 <task>
#
#   .\tasks.ps1 install
#   .\tasks.ps1 config-check
#   .\tasks.ps1 test
#   .\tasks.ps1 lint
#
# The interpreter is auto-detected: it prefers $env:PYTHON, then .\.venv, then
# any interpreter registered with the Windows `py` launcher, then `python` on
# PATH - choosing the first one that can actually `import energymind`.
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help", "install", "config-check", "data", "inject", "bench", "demo",
        "test", "lint", "fmt", "env-up", "env-down", "clean"
    )]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

function Get-PythonCandidates {
    $c = [System.Collections.Generic.List[string]]::new()
    if ($env:PYTHON) { $c.Add($env:PYTHON) }
    $venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venv) { $c.Add($venv) }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($line in (& py -0p 2>$null)) {
            if ($line -match '^\s*-V:\S+\s+\*?\s*(.+python\.exe)\s*$') {
                $c.Add($Matches[1].Trim())
            }
        }
    }
    $c.Add("python")
    return $c
}

function Test-HasPackage {
    param([string]$Exe)
    try {
        & $Exe -c "import energymind" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Resolve-Python {
    $existing = Get-PythonCandidates |
        Where-Object { $_ -eq "python" -or (Test-Path $_) } |
        Select-Object -Unique
    foreach ($exe in $existing) {
        if (Test-HasPackage $exe) { return $exe }
    }
    # Nothing has the package yet (e.g. before `install`): use the first valid one.
    return ($existing | Select-Object -First 1)
}

$python = Resolve-Python
Write-Host "python: $python" -ForegroundColor DarkGray

function Invoke-Py { param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args) & $python @Args }

switch ($Task) {
    "help" {
        @"
EnergyMind tasks:
  install       pip install -e .[dev]
  config-check  validate all configs/*.yaml
  data          load, clean, quality mask, features
  inject        generate the anomaly dataset and labels
  bench         run the benchmark, write results/runs.jsonl
  demo          ~20 cases through agents, confidence, routing, audit log
  test          run unit tests
  lint          ruff check + format check
  fmt           ruff autofix + format
  env-up        docker compose up -d (Postgres + pgvector)
  env-down      docker compose down
  clean         remove build/test caches
"@
    }
    "install" { Invoke-Py -m pip install -e ".[dev]" }
    "config-check" { Invoke-Py -m energymind.config }
    "env-up" { docker compose up -d }
    "env-down" { docker compose down }
    "data" { Invoke-Py -m energymind.pipeline data }
    "inject" { Invoke-Py -m energymind.pipeline inject }
    "bench" { Invoke-Py -m energymind.pipeline bench }
    "demo" { Invoke-Py -m energymind.pipeline demo }
    "test" { Invoke-Py -m pytest }
    "lint" {
        Invoke-Py -m ruff check src tests
        Invoke-Py -m ruff format --check src tests
    }
    "fmt" {
        Invoke-Py -m ruff check --fix src tests
        Invoke-Py -m ruff format src tests
    }
    "clean" {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            build, dist, .pytest_cache, .ruff_cache, .coverage, htmlcov
    }
}
