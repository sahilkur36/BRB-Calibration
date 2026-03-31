$ErrorActionPreference = "Stop"

# Run the full pipeline. By default all output goes to the terminal only.
#
# To mirror the same output to a log file *and* keep it on the screen, set PIPELINE_LOG:
#   $env:PIPELINE_LOG = "pipeline_log.txt"; .\run.ps1
# A footer with end time + elapsed duration is printed to the console and appended to the log.
#
# Do not use .\run.ps1 *> file if you want live progress -- that hides the terminal.
#
# Note: multi-line python commands must be one line each (PowerShell backtick continuation
# breaks if a blank line appears between lines).
#
# Logging: Start-Transcript logs PowerShell streams, not native process stdio, so Python's
# print output is often missing or incomplete in the file. When PIPELINE_LOG is set we run
# each Python step through cmd (stderr merged into stdout), replace the log at the start of
# each run, then append lines with a single shared FileStream per step (Add-Content per line
# fights locks if the log is open in an editor -- "Stream was not readable").

# J_feat: amplitude-based cycle weights w_c (--amplitude-weights on calibration Python steps).
# $false = uniform w_c=1 (default). Set $true to match amplitude weighting across this pipeline run.
$UseAmplitudeWeights = $false
$script:AmplitudeWeightPyArgs = @()
if ($UseAmplitudeWeights) {
  $script:AmplitudeWeightPyArgs = @('--amplitude-weights')
}

$script:PipelineLogFile = $null
if ($env:PIPELINE_LOG) {
  $lp = $env:PIPELINE_LOG
  if (-not [System.IO.Path]::IsPathRooted($lp)) {
    $lp = Join-Path (Get-Location).Path $lp
  }
  $script:PipelineLogFile = $lp
  $env:PYTHONUNBUFFERED = "1"
}

function Invoke-CmdArg([string]$Text) {
  if ($Text -match '[\s"]') {
    '"' + ($Text.Replace('"', '""')) + '"'
  } else {
    $Text
  }
}

function Invoke-Py {
  param(
    [Parameter(Mandatory, ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
  )
  $exe = (Get-Command python -ErrorAction Stop).Source
  if (-not $script:PipelineLogFile) {
    & $exe @PyArgs
    if ($LASTEXITCODE -ne 0) {
      throw "python failed (exit code $LASTEXITCODE)"
    }
    return
  }
  $tokens = @((Invoke-CmdArg $exe)) + ($PyArgs | ForEach-Object { Invoke-CmdArg $_ })
  # Inside cmd: merge Python stderr into stdout so PowerShell does not wrap stderr as ErrorRecords.
  $cmdLine = ($tokens -join ' ') + ' 2>&1'
  $logPath = $script:PipelineLogFile
  $utf8 = [System.Text.UTF8Encoding]::new($false)
  $fs = [System.IO.FileStream]::new(
    $logPath,
    [System.IO.FileMode]::Append,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::ReadWrite
  )
  $sw = [System.IO.StreamWriter]::new($fs, $utf8)
  try {
    cmd.exe /c $cmdLine | ForEach-Object {
      $line = if ($null -eq $_) { "" } elseif ($_ -is [string]) { $_ } else { "$_" }
      Write-Host $line
      $sw.WriteLine($line)
      $sw.Flush()
    }
  } finally {
    if ($null -ne $sw) { $sw.Dispose() }
    if ($null -ne $fs) { $fs.Dispose() }
  }
  # Prefer checking cmd's exit code (python's) here; if your host shows $LASTEXITCODE wrong after a pipe,
  # run the same step with `python ...` without PIPELINE_LOG or use PowerShell 7+.
  if ($LASTEXITCODE -ne 0) {
    throw "python failed (exit code $LASTEXITCODE)"
  }
}

function Write-PipelineFooter {
  param(
    [Parameter(Mandatory)]
    [datetime]$StartTime
  )
  $endTime = Get-Date
  $elapsed = $endTime - $StartTime
  $endStamp = $endTime.ToString("yyyy-MM-dd HH:mm:ss K")
  $h = [int][Math]::Floor($elapsed.TotalHours)
  $m = $elapsed.Minutes
  $s = $elapsed.Seconds
  $elapsedStr = if ($h -ge 1) {
    "${h}h ${m}m ${s}s"
  } elseif ($m -ge 1) {
    "${m}m ${s}s"
  } else {
    "${s}s"
  }
  $lines = @(
    ""
    "========================================================================"
    "  BRB-Calibration pipeline finished"
    "  $endStamp"
    "  Elapsed:  $elapsedStr"
    "========================================================================"
    ""
  )
  foreach ($ln in $lines) {
    Write-Output $ln
  }
  if ($script:PipelineLogFile) {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $fs = [System.IO.FileStream]::new(
      $script:PipelineLogFile,
      [System.IO.FileMode]::Append,
      [System.IO.FileAccess]::Write,
      [System.IO.FileShare]::ReadWrite
    )
    $sw = [System.IO.StreamWriter]::new($fs, $utf8)
    try {
      foreach ($ln in $lines) {
        $sw.WriteLine($ln)
      }
      $sw.Flush()
    } finally {
      if ($null -ne $sw) { $sw.Dispose() }
      if ($null -ne $fs) { $fs.Dispose() }
    }
  }
}

function Invoke-Pipeline {
  $pipelineStart = Get-Date
  $pipeLogStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
  if ($script:PipelineLogFile) {
    @(
      ""
      "========================================================================"
      "  BRB-Calibration pipeline"
      "  $pipeLogStamp"
      "========================================================================"
      ""
    ) | Set-Content -LiteralPath $script:PipelineLogFile -Encoding utf8
  }
  Write-Output ""
  Write-Output "========================================================================"
  Write-Output "  BRB-Calibration pipeline"
  Write-Output "  $pipeLogStamp"
  Write-Output "========================================================================"
  Write-Output ""

  try {
  # Echo ``config/calibration/*.csv`` heads (same paths as ``calibration_paths.py``).
  Invoke-Py scripts/calibrate/print_calibration_config_heads.py

  # Full reset (optional): uncomment to wipe regenerated data/results before rerun.
  # Preserves data/raw/ and config/calibration/*.csv (removes regenerated data/* pipeline dirs and all results/calibration/, including individual_optimize/initial_brb_parameters.csv)
  # & "$PSScriptRoot/clean_outputs.ps1"

  # --- Postprocess: cycle_points_original -> filtered -> resampled + cycle_points_resampled ---
  Invoke-Py scripts/postprocess/cycle_points.py --overwrite
  Invoke-Py scripts/postprocess/filter_force.py
  Invoke-Py scripts/postprocess/resample_filtered.py
  # Batch specimen plots (force-def, time histories)
  Invoke-Py scripts/postprocess/plot_specimens.py

  # --- Apparent b_n / b_p and geometry (exploratory) ---
  Invoke-Py scripts/calibrate/extract_bn_bp.py
  # Requires repo-root config/calibration/set_id_settings.csv (one row per set_id: steel overrides + b_p/b_n as number or stat keyword)
  Invoke-Py scripts/calibrate/build_initial_brb_parameters.py
  Invoke-Py scripts/calibrate/plot_b_slopes.py
  Invoke-Py scripts/calibrate/plot_b_histograms_and_scatter.py

  # --- Preset sim vs exp overlays (fixed b_p, b_n before L-BFGS; steel from set_id_settings.csv) ---
  Invoke-Py scripts/calibrate/plot_preset_overlays.py --params results/calibration/individual_optimize/initial_brb_parameters.csv
  # Alternative: --set-id-settings config/calibration/set_id_settings.csv if you want plot_preset_overlays to rebuild from catalog + settings instead of this CSV :)

  # --- SteelMPF calibration and sim vs exp overlays ---
  # L-BFGS box limits: config/calibration/params_limits.csv by default (override: --param-limits path)
  Invoke-Py scripts/calibrate/optimize_brb_mse.py @script:AmplitudeWeightPyArgs --initial-params results/calibration/individual_optimize/initial_brb_parameters.csv --output results/calibration/individual_optimize/optimized_brb_parameters.csv
  Invoke-Py scripts/calibrate/plot_params_vs_filtered.py --params results/calibration/individual_optimize/optimized_brb_parameters.csv --output-dir overlays

  # --- Individual optimal parameters vs geometry ---
  # Uses optimized_brb_parameters_metrics.csv to pick best set per specimen (min final_J_feat_raw) and plots vs geometry.
  Invoke-Py scripts/calibrate/plot_individual_optimal_params_vs_geometry.py

  # --- Correlations: optimal params vs geometry (train cohort) ---
  # Uses the same "optimal" selection as the geometry plots (min final_J_feat_raw).
  Invoke-Py scripts/calibrate/report_param_geometry_correlations.py

  # --- Averaged-parameter evaluation ---
  Invoke-Py scripts/calibrate/eval_averaged_params.py @script:AmplitudeWeightPyArgs --params results/calibration/individual_optimize/optimized_brb_parameters.csv --output-params results/calibration/averaged_optimize/averaged_brb_parameters.csv --output-metrics results/calibration/averaged_optimize/averaged_params_eval_metrics.csv --output-plots-dir results/plots/calibration/averaged_optimize/overlays

  # --- Generalized optimization + specimen-set eval --- (same config/calibration/params_limits.csv as optimize_brb_mse unless --param-limits)
  Invoke-Py scripts/calibrate/optimize_generalized_brb_mse.py @script:AmplitudeWeightPyArgs --params results/calibration/individual_optimize/optimized_brb_parameters.csv --output-params results/calibration/generalized_optimize/generalized_brb_parameters.csv --output-metrics results/calibration/generalized_optimize/generalized_params_eval_metrics.csv --output-plots-dir results/plots/calibration/generalized_optimize/overlays

  # --- Combined normalized overlays: one set{k}_combined_force_def_norm.png per set_id per method ---
  # Reads numerical-model *_simulated.csv under each method's *simulated_force/ (from optimize_brb_mse / eval above).
  Invoke-Py scripts/calibrate/plot_compare_calibration_overlays.py

  # --- Generalized overlays with train-weighted mean b_p/b_n (same steel as generalized per set_id) ---
  Invoke-Py scripts/calibrate/plot_generalized_train_mean_bn_bp_overlays.py

  # --- Optimized-parameter summary (summary_statistics/) ---
  # Writes calibration_parameter_summary.md and, next to it:
  #   *_generalized.csv / *_individual.csv  — rollup by parameter (mean/min/max [+ optimum / weighted means])
  #   *_generalized_by_set.csv / *_individual_by_set.csv — one row per set_id (wide parameter tables; generalized includes eval columns)
  Invoke-Py scripts/calibrate/report_calibration_param_tables.py --write summary_statistics/calibration_parameter_summary.md

  # --- Debug figures ---
  Invoke-Py scripts/calibrate/plot_cycle_energy_debug.py @script:AmplitudeWeightPyArgs --params results/calibration/individual_optimize/optimized_brb_parameters.csv
  # Always --amplitude-weights so *_landmarks_exp.csv w_c matches J_feat (independent of $UseAmplitudeWeights).
  Invoke-Py scripts/calibrate/plot_cycle_landmarks_debug.py --amplitude-weights --params results/calibration/individual_optimize/optimized_brb_parameters.csv

  # --- Averaged vs generalized narrative report (default: results/calibration/averaged_vs_generalized_metrics_report.md) ---
  Invoke-Py scripts/calibrate/report_averaged_vs_generalized_metrics.py

  } finally {
    Write-PipelineFooter -StartTime $pipelineStart
  }
}

Invoke-Pipeline
