# Sweep one weight (mid and end together) over factors of its compiled value and report, per point,
# the strength (match against the compiled weights) and the loss for one or more loss variants.
#
# For every factor f: candidate = compiled * f (rounded, mid and end), played by run_sharded.ps1
# (3 parallel shards); the loss is computed with TuneSGD -e on the training data.
# Finished points are skipped, so the script can be started again to continue.
#
# Example:  .\run_sweep.ps1 -Name matdiff -Params ewMaterialDiff
#           .\run_sweep.ps1 -Name king -Params ewKingThreat -Factors 0.25,0.5,1.5,2 -Pairs 4000
#           .\run_sweep.ps1 -Name pp -Params ewPassPawnLev -Variants "220:0:5","300:0:5","220:1:1"
#
# -Params: one or more weights (bare name = mid and end); all are scaled by the same factor.
# -Variants: loss variants as "k:q:L" (sigmoid scale, score weight, loss function number).
# Output: <Out>\<Name>\table.txt, the configs f*.cfg and the match results under <Root>\sw-<Name>-f*.

param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string[]]$Params,
    [double[]]$Factors = @(0.5, 0.75, 1.25, 1.5),   # 1.0 (compiled) is always added for the loss, not played
    [int]$Pairs = 2000,     # opening pairs per shard; games per point = 2 * Pairs * Shards
    [int]$Shards = 3,
    [int]$Nodes = 1000,
    [string[]]$Variants = @("220:0:5"),
    [string]$Data = "C:\Learn\Tune\cxig\data",
    [string]$Out = "C:\Learn\Tune\cxig\sweeps",
    [string]$Root = "C:\Learn\Tune\cxig\groups",
    [string]$Weights = "C:\Learn\Tune\cxig\Weights.hs",
    [string]$Tune = "C:\astra\TuneSGD\TuneSGD-tsub.exe",
    [string]$Fens = "C:\Learn\Tune\cxig\open\open-moves-uniq.fen",
    [string]$Exe = "C:\astra\SelfPlay\SelfPlay-tsub.exe"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$sharded = Join-Path $here "run_sharded.ps1"
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$noBom = New-Object System.Text.UTF8Encoding($false)

foreach ($f in @($Weights, $Tune, $Fens, $Exe, $sharded)) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "File not found: $f" }
}
if (-not (Test-Path -LiteralPath $Data -PathType Container)) { throw "Data directory not found: $Data" }

# Compiled values: lines like ("ewName", (mid, end)),
$compiled = @{}
$rx = '\("(\w+)",\s*\(\s*(-?\d+),\s*(-?\d+)\)\)'
foreach ($m in [regex]::Matches((Get-Content -LiteralPath $Weights -Raw), $rx)) {
    $compiled["mid." + $m.Groups[1].Value] = [double]$m.Groups[2].Value
    $compiled["end." + $m.Groups[1].Value] = [double]$m.Groups[3].Value
}
$bare = @($Params | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$names = @()
foreach ($p in $bare) {
    foreach ($ph in @("mid", "end")) {
        $k = if ($p -match '^(mid|end)\.') { $p } else { "$ph.$p" }
        if (-not $compiled.ContainsKey($k)) { throw "Unknown weight: $k" }
        if ($names -notcontains $k) { $names += $k }
        if ($p -match '^(mid|end)\.') { break }
    }
}

$sweepDir = Join-Path $Out $Name
New-Item -ItemType Directory -Force -Path $sweepDir | Out-Null

function Get-Loss([string]$cfg, [string]$variant) {
    $parts = $variant -split ":"
    if ($parts.Count -lt 2) { throw "Bad variant '$variant', expected k:q or k:q:L" }
    $lf = if ($parts.Count -ge 3) { $parts[2] } else { "5" }
    $a = @("-e", "-i", $Data, "-L", $lf, "-k", $parts[0], "-q", $parts[1])
    if ($cfg) { $a += @("-c", $cfg) }
    $o = & $Tune @a | Where-Object { $_ -match '^Loss:' } | Select-Object -Last 1
    if (-not $o) { throw "No loss from TuneSGD for variant $variant (config '$cfg')" }
    return [double]::Parse(($o -replace '^Loss:\s*', ''), $inv)
}

function Get-EloText([double]$w, [double]$d, [double]$l) {
    $n = $w + $d + $l
    $s = ($w + 0.5 * $d) / $n
    $var = ($w * (1 - $s) * (1 - $s) + $d * (0.5 - $s) * (0.5 - $s) + $l * $s * $s) / $n
    $ci = 1.96 * 400 / [Math]::Log(10) / ($s * (1 - $s)) * [Math]::Sqrt($var / $n)
    $elo = -400 * [Math]::Log10(1 / $s - 1)
    return ("{0,7:+0.0;-0.0} +-{1,5:0.0}" -f $elo, $ci)
}

# Points: the compiled point (factor 1, loss only) and the played factors
$points = @(1.0) + @($Factors | Where-Object { $_ -ne 1.0 })
$rows = @()
$baseLoss = @{}
foreach ($v in $Variants) { $baseLoss[$v] = Get-Loss "" $v }

foreach ($f in ($points | Sort-Object)) {
    $tag = ($f.ToString("0.###", $inv)) -replace '\.', 'p'
    $cfg = ""
    $vals = @()
    if ($f -ne 1.0) {
        $lines = @("-- sweep $Name, factor $f of the compiled values")
        foreach ($k in $names) {
            $nv = [Math]::Round($compiled[$k] * $f, 0, [MidpointRounding]::AwayFromZero)
            $vals += $nv
            $lines += "$k = $([int]$nv)"
        }
        $cfg = Join-Path $sweepDir "f$tag.cfg"
        [System.IO.File]::WriteAllText($cfg, (($lines -join "`n") + "`n"), $noBom)
        if ((@($names | ForEach-Object { $compiled[$_] }) -join ",") -eq ($vals -join ",")) {
            Write-Warning "factor $f gives the compiled values after rounding, skipped"
            continue
        }
    } else {
        $vals = @($names | ForEach-Object { $compiled[$_] })
    }

    # strength
    $eloText = "     (compiled)"
    $wdlText = ""
    if ($f -ne 1.0) {
        $gname = "sw-$Name-f$tag"
        $resFile = Join-Path (Join-Path $Root $gname) "result.txt"
        if (-not (Test-Path -LiteralPath $resFile)) {
            Write-Host "== factor $f : $($names -join ' ') = $($vals -join ' ')"
            & $sharded -Name $gname -Params ($names -join ",") -Source $cfg -Pairs $Pairs -Shards $Shards `
                -Nodes $Nodes -Root $Root -Fens $Fens -Exe $Exe | Out-Null
        } else {
            Write-Host "== factor $f : result exists, skipped"
        }
        $line = Select-String -LiteralPath $resFile -Pattern '^WDL:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)' | Select-Object -First 1
        if (-not $line) { throw "No WDL in $resFile" }
        $g = $line.Matches[0].Groups
        $eloText = Get-EloText ([double]$g[1].Value) ([double]$g[2].Value) ([double]$g[3].Value)
        $wdlText = "$($g[1].Value)/$($g[2].Value)/$($g[3].Value)"
    }

    # loss for every variant
    $lossText = @()
    foreach ($v in $Variants) {
        $loss = if ($f -eq 1.0) { $baseLoss[$v] } else { Get-Loss $cfg $v }
        $rel = 100 * ($loss - $baseLoss[$v]) / $baseLoss[$v]
        $lossText += ("{0:0.0} ({1:+0.000;-0.000}%)" -f $loss, $rel)
    }
    $rows += [pscustomobject]@{ F = $f; Vals = ($vals -join ","); Elo = $eloText; Wdl = $wdlText; Loss = ($lossText -join " | ") }
    Write-Host ("f={0,-5} values {1,-18} Elo {2}   loss {3}" -f $f, ($vals -join ","), $eloText, ($lossText -join " | "))
}

$header = @(
    "sweep $Name  weights: $($names -join ' ')",
    "games per point: $(2 * $Pairs * $Shards) at $Nodes nodes; loss variants (k:q:L): $($Variants -join '  ')",
    "factor values              Elo vs compiled        WDL                   loss (change vs compiled)"
)
$table = $rows | Sort-Object F | ForEach-Object {
    "{0,-6} {1,-18} {2,-20} {3,-22} {4}" -f $_.F, $_.Vals, $_.Elo, $_.Wdl, $_.Loss
}
$all = $header + $table
[System.IO.File]::WriteAllText((Join-Path $sweepDir "table.txt"), (($all -join "`n") + "`n"), $noBom)
Write-Host ""
$all | ForEach-Object { Write-Host $_ }
