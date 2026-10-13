# Add up the WDL of several group runs (e.g. shards of one test run in parallel) and report the Elo.
#
# Example:  .\merge_groups.ps1 -Groups rook75-s1,rook75-s2,rook75-s3
# The groups are read from <Root>\<Group>\result.txt (line "WDL:  w / d / l", candidate's point of view).
# Use -Name to also write the merged summary to <Root>\<Name>\result.txt.

param(
    [Parameter(Mandatory = $true)]
    [string[]]$Groups,
    [string]$Root = "C:\Learn\Tune\cxig\groups",
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"
$noBom = New-Object System.Text.UTF8Encoding($false)

$w = 0.0; $d = 0.0; $l = 0.0
foreach ($g in ($Groups | ForEach-Object { $_ -split "," } | Where-Object { $_ })) {
    $f = Join-Path (Join-Path $Root $g) "result.txt"
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "No result yet: $f" }
    $line = Select-String -LiteralPath $f -Pattern '^WDL:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)' | Select-Object -First 1
    if (-not $line) { throw "No WDL line in $f" }
    $m = $line.Matches[0].Groups
    $w += [double]$m[1].Value; $d += [double]$m[2].Value; $l += [double]$m[3].Value
    Write-Host ("{0,-20} {1} / {2} / {3}" -f $g, $m[1].Value, $m[2].Value, $m[3].Value)
}
$n = $w + $d + $l
$s = ($w + 0.5 * $d) / $n
$var = ($w * (1 - $s) * (1 - $s) + $d * (0.5 - $s) * (0.5 - $s) + $l * $s * $s) / $n
$se = [Math]::Sqrt($var / $n)
$elo = -400 * [Math]::Log10(1 / $s - 1)
$ci = 1.96 * 400 / [Math]::Log(10) / ($s * (1 - $s)) * $se

$summary = @(
    "merged:   $($Groups -join ', ')",
    "games:    $([int]$n)",
    "WDL:      $([int]$w) / $([int]$d) / $([int]$l)",
    ("score:    {0:P1}" -f $s),
    ("Elo:      {0:+0.0;-0.0} +- {1:N1} (95%, game level)" -f $elo, $ci)
)
$summary | ForEach-Object { Write-Host $_ }
if ($Name) {
    $dir = Join-Path $Root $Name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $dir "result.txt"), (($summary -join "`n") + "`n"), $noBom)
}
