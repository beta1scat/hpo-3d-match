# PowerShell script to run 3-seed HPO experiments for Chapter 3
# Seeds: 42 (default baseline, skips if completed), 43, 44
# Models: bracket_planar, screw_black, star
# Concurrency: Starts all 3 models in PARALLEL for each seed by default!
# Objective: lexicographical-recall-first, Sampler: TPE, Pruner: Nop, Budget: 500

param(
    [switch]$NoUv,
    [string]$PythonExe = "",
    [switch]$Serial  # Run sequentially instead of 3 models in parallel per seed
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$UseUv = $false
if (-not $NoUv -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    $UseUv = $true
    Write-Host "[*] Execution Environment: uv (uv run python)" -ForegroundColor Green
} else {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = (Get-Command python).Source
    }
    Write-Host "[*] Execution Environment: direct python ($PythonExe)" -ForegroundColor Cyan
}

Write-Host "[*] Working directory: $ScriptDir" -ForegroundColor Cyan
Write-Host "[*] Execution mode: $(if ($Serial) { 'Serial (1 model at a time)' } else { 'Parallel (3 models per seed concurrently)' })" -ForegroundColor Cyan

$Models = @("bracket_planar", "screw_black", "star")
$Seeds = @(42, 43, 44)
$Budget = 500
$Sampler = "TPE"
$Pruner = "Nop"

$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$GlobalLog = Join-Path $LogDir "multi_seed_hpo_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "[*] Master execution log: $GlobalLog" -ForegroundColor Cyan

foreach ($Seed in $Seeds) {
    Write-Host "`n================================================================================" -ForegroundColor Yellow
    Write-Host ">>> Starting Batch for Seed: $Seed (Models: $($Models -join ', '))" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Yellow
    Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] START_SEED $Seed"

    $Jobs = @()

    foreach ($Model in $Models) {
        $ExpDirName = "${Model}_tpe_nop_lexrecall_b${Budget}_s${Seed}"
        $ExpDir = Join-Path $ScriptDir (Join-Path "results" $ExpDirName)
        $SummaryFile = Join-Path $ExpDir "experiment_summary.json"
        $ModelLog = Join-Path $LogDir "${ExpDirName}.log"

        if (Test-Path $SummaryFile) {
            Write-Host "[SKIP] Model=$Model Seed=$Seed ($ExpDirName) already completed." -ForegroundColor Green
            Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] SKIP $ExpDirName (Already completed)"
            continue
        }

        if ($Serial) {
            Write-Host "[EXEC-SERIAL] Model=$Model Seed=$Seed (Log: $ModelLog)" -ForegroundColor White
            Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] START $ExpDirName (Serial)"
            $StartTime = Get-Date
            $ArgsList = @(
                "run_experiment.py",
                "--model", $Model,
                "--sampler", $Sampler,
                "--pruner", $Pruner,
                "--budget", $Budget,
                "--seed", $Seed
            )
            if ($UseUv) {
                & uv run python @ArgsList 2>&1 | Tee-Object -FilePath $ModelLog
            } else {
                & $PythonExe @ArgsList 2>&1 | Tee-Object -FilePath $ModelLog
            }
            $ExitCode = $LASTEXITCODE
            $Dur = (Get-Date) - $StartTime
            if ($ExitCode -ne 0) {
                Write-Host "[ERROR] $ExpDirName failed with exit code $ExitCode after $($Dur.TotalMinutes.ToString('F2'))m." -ForegroundColor Red
                Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FAILED $ExpDirName code=$ExitCode"
                exit $ExitCode
            } else {
                Write-Host "[SUCCESS] $ExpDirName completed in $($Dur.TotalMinutes.ToString('F2'))m." -ForegroundColor Green
                Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FINISHED $ExpDirName in $($Dur.TotalMinutes.ToString('F2'))m"
            }
        } else {
            # Start background job for parallel execution
            Write-Host "[LAUNCH-PARALLEL] Model=$Model Seed=$Seed (Log: $ModelLog)" -ForegroundColor White
            Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] LAUNCH $ExpDirName (Parallel)"

            $job = Start-Job -ScriptBlock {
                param($sDir, $m, $smp, $prn, $b, $s, $uvFlag, $pyPath, $mLog)
                Set-Location $sDir
                $args = @(
                    "run_experiment.py",
                    "--model", $m,
                    "--sampler", $smp,
                    "--pruner", $prn,
                    "--budget", $b,
                    "--seed", $s
                )
                if ($uvFlag) {
                    & uv run python @args 2>&1 | Out-File -FilePath $mLog -Encoding utf8
                } else {
                    & $pyPath @args 2>&1 | Out-File -FilePath $mLog -Encoding utf8
                }
                return $LASTEXITCODE
            } -ArgumentList $ScriptDir, $Model, $Sampler, $Pruner, $Budget, $Seed, $UseUv, $PythonExe, $ModelLog

            $Jobs += [PSCustomObject]@{
                Job = $job
                Model = $Model
                Seed = $Seed
                ExpDirName = $ExpDirName
                ModelLog = $ModelLog
                StartTime = (Get-Date)
            }
        }
    }

    # If running in parallel, wait for all models of this seed to complete
    if (-not $Serial -and $Jobs.Count -gt 0) {
        Write-Host "`n[*] $($Jobs.Count) models running in parallel for Seed $Seed. Waiting for all to complete..." -ForegroundColor Cyan

        while ($true) {
            $running = $Jobs | Where-Object { $_.Job.State -eq "Running" }
            if ($running.Count -eq 0) {
                break
            }
            Start-Sleep -Seconds 10
            $progressReport = ($Jobs | ForEach-Object {
                $elapsed = [math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1)
                "$($_.Model): $($_.Job.State) (${elapsed}m)"
            }) -join " | "
            Write-Host "  -> [Seed $Seed Progress] $progressReport" -ForegroundColor DarkGray
        }

        # Check job completion and exit codes
        $hasError = $false
        foreach ($item in $Jobs) {
            $exitCode = Receive-Job -Job $item.Job
            Remove-Job -Job $item.Job
            $dur = [math]::Round(((Get-Date) - $item.StartTime).TotalMinutes, 2)

            if ($exitCode -eq 0) {
                Write-Host "[SUCCESS] Model=$($item.Model) Seed=$($item.Seed) completed in ${dur} minutes." -ForegroundColor Green
                Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FINISHED $($item.ExpDirName) in ${dur}m"
            } else {
                Write-Host "[ERROR] Model=$($item.Model) Seed=$($item.Seed) failed with exit code $exitCode after ${dur}m. See: $($item.ModelLog)" -ForegroundColor Red
                Add-Content -Path $GlobalLog -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FAILED $($item.ExpDirName) exit=$exitCode"
                $hasError = $true
            }
        }

        if ($hasError) {
            Write-Host "`n[FATAL] One or more models failed for Seed $Seed. Halting batch execution." -ForegroundColor Red
            exit 1
        }
        Write-Host "[*] Batch for Seed $Seed successfully completed!`n" -ForegroundColor Green
    }
}

Write-Host "`n[*] All seeds finished. Generating multi-seed statistical summary..." -ForegroundColor Cyan
if ($UseUv) {
    & uv run python "summarize_multi_seed_hpo.py" --output-markdown "MULTI_SEED_HPO_SUMMARY.md" --export-csv "multi_seed_hpo_trials.csv"
} else {
    & $PythonExe "summarize_multi_seed_hpo.py" --output-markdown "MULTI_SEED_HPO_SUMMARY.md" --export-csv "multi_seed_hpo_trials.csv"
}
Write-Host "[*] Done. All multi-seed experiments and statistical tables generated." -ForegroundColor Green
