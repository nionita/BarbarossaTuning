# Measure the strength effect of one group of parameters from a tuned config.
#
# The candidate is the compiled weights (or -Base) with only the listed parameters taken from -Source.
# With -Scale the change of those parameters is scaled: value = compiled + Scale * (source - compiled),
# e.g. -Scale 0.5 for half of the tuned change, -Scale -1 for the mirrored change.
# Use another -Group name for every variant (e.g. rook-m05 for -Scale -0.5): an existing group
# directory is never overwritten without -Force.
# It plays a fixed number of opening pairs against the base (no early stop) and the Elo is reported.
# Everything is created under <Root>\<Group>\ :
#   candidate.cfg      the partial config: only the group's parameters (with -Config: a copy of that file)
#   match\             working directory of SelfPlay
#   match.stdout.txt   full SelfPlay output
#   result.txt         summary: parameters, WDL, score, Elo with 95% interval
#
# Example:  .\run_group.ps1 -Group king -Params ewKingSafe,ewKingOpen,ewKingPlaceCent,ewKingPlacePwns,ewKingPawn,ewKingThreat,ewKingProtB,ewKingProtN
#           (a bare name selects mid. and end.; mid.ewX / end.ewX select one value)
#           .\run_group.ps1 -Group pst40 -Config C:\Learn\Tune\hvar\cand\pst40.cfg
#           (-Config: a complete candidate config, played as it is, e.g. with PST lines from hv_cfg.py)

param(
    [Parameter(Mandatory = $true)]
    [string]$Group,
    [string[]]$Params = @(),
    [string]$Source = "C:\Learn\Tune\cxig\run3\tuned.cfg",
    [string]$Config = "",   # complete candidate config, played as it is (instead of -Params, -Source, -Scale)
    [string]$DataFile = "", # SelfPlay --data: leaf positions of every move with the game result (and .meta)
    [double]$Scale = 1,     # scale of the tuned change relative to the compiled weights; 1 = as in -Source
    [string]$Weights = "C:\Learn\Tune\cxig\Weights.hs", # compiled values
    [string]$Base = "",     # base config; empty = compiled weights of the SelfPlay exe (--base-current)
    [int]$Pairs = 3000,     # opening pairs to play (-f); the games are 2 * Pairs
    [int]$Nodes = 1000,     # nodes per move (-n)
    [int]$Skip = 0,         # openings to skip (-s); use i * Pairs for shard i, so parallel runs play different openings
    [string]$Fens = "C:\Learn\Tune\cxig\open\open-moves-uniq.fen",
    [string]$Root = "C:\Learn\Tune\cxig\groups",
    [string]$Exe = "C:\astra\SelfPlay\SelfPlay-tsub.exe",
    [switch]$Force          # overwrite an existing group directory
)

$ErrorActionPreference = "Stop"

if (-not $Config -and $Params.Count -eq 0) { throw "Give -Params or -Config" }
$needed = if ($Config) { @($Config, $Fens, $Exe) } else { @($Source, $Fens, $Exe, $Weights) }
foreach ($f in $needed) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "File not found: $f" }
}
if ($Base -and -not (Test-Path -LiteralPath $Base -PathType Leaf)) { throw "Base config not found: $Base" }

$groupDir = Join-Path $Root $Group
$matchDir = Join-Path $groupDir "match"
$candCfg  = Join-Path $groupDir "candidate.cfg"
$logFile  = Join-Path $groupDir "match.stdout.txt"
$resFile  = Join-Path $groupDir "result.txt"
$noBom    = New-Object System.Text.UTF8Encoding($false)

if ((Test-Path -LiteralPath $groupDir) -and -not $Force) {
    throw "$groupDir exists already; use another group name or -Force"
}

if ($Config) {
# The whole config: its value names (weights mid./end., PST pst., eval parameters) for the summary
$wanted = @(Get-Content -LiteralPath $Config |
    ForEach-Object { if ($_ -match '^\s*([A-Za-z][\w.]*)\s*=') { $Matches[1] } })
if ($wanted.Count -eq 0) { throw "No values found in $Config" }
New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
$cfgLines = @("-- group ${Group}: config $Config") + @(Get-Content -LiteralPath $Config)
[System.IO.File]::WriteAllText($candCfg, (($cfgLines -join "`n") + "`n"), $noBom)
} else {
# The wanted names: a bare name selects the mid and the end value
$wanted = @()
foreach ($p in ($Params | ForEach-Object { $_ -split "," } | Where-Object { $_ })) {
    if ($p -match '^(mid|end)\.') { $wanted += $p } else { $wanted += "mid.$p"; $wanted += "end.$p" }
}

# Take their values from the source config (lines like "mid.ewKingPawn = 19")
$selected = @{}
foreach ($line in (Get-Content -LiteralPath $Source)) {
    if ($line -match '^\s*((?:mid|end)\.\w+)\s*=\s*(-?[0-9.eE+-]+)\s*$') {
        if ($wanted -contains $Matches[1]) { $selected[$Matches[1]] = $Matches[2] }
    }
}
$missing = @($wanted | Where-Object { -not $selected.ContainsKey($_) })
if ($missing.Count -gt 0) { throw "Not found in ${Source}: $($missing -join ', ')" }

$inv = [System.Globalization.CultureInfo]::InvariantCulture
if ($Scale -ne 1) {
    # compiled values from the weights table lines like: ("ewName", (mid, end)),
    $compiled = @{}
    $rx = '\("(\w+)",\s*\(\s*(-?\d+),\s*(-?\d+)\)\)'
    foreach ($m in [regex]::Matches((Get-Content -LiteralPath $Weights -Raw), $rx)) {
        $compiled["mid." + $m.Groups[1].Value] = [double]$m.Groups[2].Value
        $compiled["end." + $m.Groups[1].Value] = [double]$m.Groups[3].Value
    }
    foreach ($k in @($selected.Keys)) {
        if (-not $compiled.ContainsKey($k)) { throw "No compiled value for $k in $Weights" }
        $v = [double]::Parse($selected[$k], $inv)
        $nv = $compiled[$k] + $Scale * ($v - $compiled[$k])
        $selected[$k] = ([Math]::Round($nv, 0, [MidpointRounding]::AwayFromZero)).ToString($inv)
    }
}

New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
$cfgLines = @("-- group $Group from $Source, scale $Scale") + ($wanted | ForEach-Object { "$_ = $($selected[$_])" })
[System.IO.File]::WriteAllText($candCfg, (($cfgLines -join "`n") + "`n"), $noBom)
}

$spArgs = @("-m", $matchDir, "-a", $candCfg)
if ($Base) { $spArgs += @("-b", $Base) } else { $spArgs += "--base-current" }
$spArgs += @("-i", $Fens, "-f", $Pairs, "-n", $Nodes)
if ($Skip -gt 0) { $spArgs += @("-s", $Skip) }
if ($DataFile) { $spArgs += @("--data", [System.IO.Path]::GetFullPath($DataFile)) }

Write-Host "$Exe $($spArgs -join ' ')"
Write-Host "Group ${Group}: $($wanted.Count) values"
$start = Get-Date
[System.IO.File]::WriteAllText($logFile, "")
# stderr of SelfPlay goes into the log too (error messages of a crashing shard); in PowerShell 5.1 the
# redirected stderr lines are error records, which would stop the script with ErrorActionPreference Stop
$eap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Exe @spArgs 2>&1 | ForEach-Object {
    $line = "$_"
    Add-Content -LiteralPath $logFile -Value $line -Encoding utf8
    if ($line -match '^(Games|WDL):') { Write-Host $line }
}
$code = $LASTEXITCODE
$ErrorActionPreference = $eap
$minutes = ((Get-Date) - $start).TotalMinutes
if ($code -ne 0) {
    $failText = "SelfPlay failed with exit code $code after $('{0:N1}' -f $minutes) minutes, see $logFile"
    [System.IO.File]::WriteAllText((Join-Path $groupDir "failed.txt"), $failText + "`n", $noBom)
    throw $failText
}

# The last WDL line is from the candidate's point of view: (wins, draws, losses)
$wdl = Select-String -LiteralPath $logFile -Pattern '(?:WDL|End result):\s*\((\d+),\s*(\d+),\s*(\d+)\)' |
       Select-Object -Last 1
if (-not $wdl) { throw "No WDL line found in $logFile" }
$w = [double]$wdl.Matches[0].Groups[1].Value
$d = [double]$wdl.Matches[0].Groups[2].Value
$l = [double]$wdl.Matches[0].Groups[3].Value
$n = $w + $d + $l
$s = ($w + 0.5 * $d) / $n
$var = ($w * (1 - $s) * (1 - $s) + $d * (0.5 - $s) * (0.5 - $s) + $l * $s * $s) / $n
$se = [Math]::Sqrt($var / $n)
$elo = -400 * [Math]::Log10(1 / $s - 1)
$ci = 1.96 * 400 / [Math]::Log(10) / ($s * (1 - $s)) * $se

$summary = @(
    "group:    $Group",
    $(if ($Config) { "config:   $Config ($($wanted.Count) values)" } else { "params:   $($wanted -join ' ')" }),
    $(if ($Config) { "source:   -" } else { "source:   $Source" }),
    "scale:    $Scale (value = compiled + scale * (source - compiled))",
    "base:     $(if ($Base) { $Base } else { 'compiled weights' })",
    "games:    $([int]$n) ($Pairs pairs from opening $Skip, $Nodes nodes per move), $('{0:N1}' -f $minutes) minutes",
    "WDL:      $([int]$w) / $([int]$d) / $([int]$l)",
    ("score:    {0:P1}" -f $s),
    ("Elo:      {0:+0.0;-0.0} +- {1:N1} (95%, game level)" -f $elo, $ci)
)
[System.IO.File]::WriteAllText($resFile, (($summary -join "`n") + "`n"), $noBom)
$summary | ForEach-Object { Write-Host $_ }
