# Run one candidate (a parameter group, optionally scaled) as several parallel shards and merge the result.
# Each shard is one run_group.ps1 on its own range of openings (-Skip = shard * Pairs), so the shards
# never play the same opening. Total games = 2 * Pairs * Shards.
#
# Examples:
#   .\run_sharded.ps1 -Name all-h -All -Scale 0.5             # the whole tuned config at half step
#   .\run_sharded.ps1 -Name rook75 -Params ewRookHOpen        # one weight; bare name = mid and end
#   .\run_sharded.ps1 -Name rookopen75 -Params mid.ewRookHOpen -Source C:\Learn\Tune\cxig\recov-up\tuned.cfg
#   .\run_sharded.ps1 -Name pst40 -Config C:\Learn\Tune\hvar\cand\pst40.cfg   # a complete config as it is
#   .\run_sharded.ps1 -Name gen1 -Config ...\compiled.cfg -DataDir C:\Learn\Tune\hvar\sp-data   # with data collection
# -DataDir: every shard writes the leaf positions of its games with the result to <DataDir>\<Name>-s<i>-filt.csv
# (and .meta, see SelfPlay --data); -Offset: first opening of shard 1 (other runs on other openings)
# Results: <Root>\<Name>-s1.. <Name>-sN (shards) and <Root>\<Name> (merged result.txt).

param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [string[]]$Params = @(),
    [switch]$All,           # all weights of the source config (every value in it)
    [string]$Config = "",   # complete candidate config, played as it is (instead of -Params / -All)
    [string]$DataDir = "",  # collect the leaf positions with the game results there (SelfPlay --data)
    [int]$Offset = 0,       # openings skipped before shard 1
    [double]$Scale = 1,
    [int]$Pairs = 7000,     # opening pairs per shard
    [int]$Shards = 3,
    [int]$Nodes = 1000,
    [string]$Source = "C:\Learn\Tune\cxig\run3\tuned.cfg",
    [string]$Root = "C:\Learn\Tune\cxig\groups",
    [string]$Fens = "C:\Learn\Tune\cxig\open\open-moves-uniq.fen",
    [string]$Exe = "C:\astra\SelfPlay\SelfPlay-tsub.exe"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$group = Join-Path $here "run_group.ps1"
$merge = Join-Path $here "merge_groups.ps1"

if (-not $Config -and -not $All -and $Params.Count -eq 0) { throw "Give -Params, -All or -Config" }
if ($Config) {
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Config not found: $Config" }
    $names = @(Get-Content -LiteralPath $Config |
        ForEach-Object { if ($_ -match '^\s*([A-Za-z][\w.]*)\s*=') { $Matches[1] } })
} elseif ($All) {
    $names = @(Get-Content -LiteralPath $Source |
        ForEach-Object { if ($_ -match '^\s*((?:mid|end)\.\w+)\s*=') { $Matches[1] } })
    if ($names.Count -eq 0) { throw "No values found in $Source" }
} else {
    $names = @($Params | ForEach-Object { $_ -split "," } | Where-Object { $_ })
}
$paramText = $names -join ","

$fenCount = @(Get-Content -LiteralPath $Fens).Count
if ($Offset + $Pairs * $Shards -gt $fenCount) {
    throw "$Offset + $Pairs pairs x $Shards shards = $($Offset + $Pairs * $Shards) > $fenCount openings"
}
if ($DataDir) { New-Item -ItemType Directory -Force -Path $DataDir | Out-Null }
foreach ($i in 1..$Shards) {
    if (Test-Path -LiteralPath (Join-Path $Root "$Name-s$i")) { throw "$Name-s$i exists already; use another -Name" }
}

$procs = @()
foreach ($i in 1..$Shards) {
    $argList = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $group,
        "-Group", "$Name-s$i", "-Pairs", $Pairs, "-Skip", ($Offset + ($i - 1) * $Pairs), "-Nodes", $Nodes,
        "-Root", $Root, "-Fens", $Fens, "-Exe", $Exe
    )
    if ($DataDir) { $argList += @("-DataFile", (Join-Path $DataDir "$Name-s$i-filt.csv")) }
    if ($Config) {
        $argList += @("-Config", $Config)
    } else {
        $argList += @("-Params", $paramText, "-Source", $Source, "-Scale", $Scale)
    }
    $procs += Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WindowStyle Minimized -PassThru
    Write-Host "started shard $i (openings $($Offset + ($i - 1) * $Pairs) .. $($Offset + $i * $Pairs - 1)), pid $($procs[-1].Id)"
}
Write-Host "waiting for $Shards shards, $($names.Count) values, scale $Scale ..."
$procs | Wait-Process

& $merge -Groups (1..$Shards | ForEach-Object { "$Name-s$_" }) -Root $Root -Name $Name
