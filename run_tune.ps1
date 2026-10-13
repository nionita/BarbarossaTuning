# Run one TuneSGD optimization on the cxig dataset.
# The run directory must exist; it receives tuned.cfg, tune.log and run.txt (the settings).
#
# Example:  .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\run1
#           .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\run2 -K 220 -Q 0 -Fix -1
#           .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\run4 -K 220 -Q 0 -Fix -1 -Config C:\Learn\Tune\cxig\run3\tuned.cfg
#           .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\run5 -Params ewMobilityKnight,end.ewBishopPair -Optim 40

param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [double]$K = 220,       # sigmoid scale in centipawns (TuneSGD -k); the best fit for the results is about 210-220
    [double]$Q = 0,         # weight of the SCORE in the loss (-q): 0 = game result only (Texel), 1 = score only
    [int]$Optim = 92,       # dimension steps (-O); 92 = one full round over all weights
    [int]$Loss = 5,         # loss function (-L); 5 = score/result loss
    [int]$Fix = 0,          # dimension index not modified (-F); 0 = mid.ewMaterialDiff, -1 = none
    [string[]]$Params = @(), # parameters to optimize (-p), e.g. ewBishopPair,mid.ewKingSafe; empty = all
                            # a bare name selects mid and end; -Optim counts steps (dimensions tried)
    [string]$Config = "",   # start from the values of this config file (-c), e.g. a previous tuned.cfg
    [string]$Data = "C:\Learn\Tune\cxig\data",
    [string]$Exe = "C:\astra\TuneSGD\TuneSGD-tsub.exe",
    [switch]$Force          # allow overwriting an existing tuned.cfg in the run directory
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) { throw "Run directory not found: $RunDir" }
if (-not (Test-Path -LiteralPath $Data   -PathType Container)) { throw "Data directory not found: $Data" }
if (-not (Test-Path -LiteralPath $Exe    -PathType Leaf))      { throw "Executable not found: $Exe" }
if ($Config -and -not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Config file not found: $Config" }

$cfg  = Join-Path $RunDir "tuned.cfg"
$log  = Join-Path $RunDir "tune.log"
$info = Join-Path $RunDir "run.txt"
$inv  = [System.Globalization.CultureInfo]::InvariantCulture

if ((Test-Path -LiteralPath $cfg) -and -not $Force) {
    throw "$cfg exists already; use another run directory or -Force"
}

$tuneArgs = @(
    "-i", $Data,
    "-o", $cfg,
    "-O", $Optim,
    "-F", $Fix,
    "-L", $Loss,
    "-k", $K.ToString($inv),
    "-q", $Q.ToString($inv)
)
$paramList = ($Params | Where-Object { $_ }) -join ","
if ($paramList) { $tuneArgs += @("-p", $paramList) }
if ($Config)    { $tuneArgs += @("-c", $Config) }

$cmdLine = "$Exe $($tuneArgs -join ' ')"
Write-Host $cmdLine
$start = Get-Date
Set-Content -LiteralPath $info -Encoding utf8 -Value @(
    "started: $($start.ToString('yyyy-MM-dd HH:mm:ss'))",
    "command: $cmdLine"
)

[System.IO.File]::WriteAllText($log, "")
& $Exe @tuneArgs | ForEach-Object {
    Write-Host $_
    Add-Content -LiteralPath $log -Value $_ -Encoding utf8
}
$code = $LASTEXITCODE
$minutes = ((Get-Date) - $start).TotalMinutes
Add-Content -LiteralPath $info -Encoding utf8 -Value ("finished: {0:N1} minutes, exit code {1}" -f $minutes, $code)
Write-Host ("Finished in {0:N1} minutes, exit code {1}" -f $minutes, $code)
if ($code -ne 0) { exit $code }
if (Test-Path -LiteralPath $cfg) { Write-Host "Config: $cfg" } else { Write-Warning "No config written (no improvement found?)" }
