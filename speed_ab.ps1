# Speed comparison of two SelfPlay builds with the same eval (deterministic perf builds: same tree, same node
# counts), alternating in random order; reports nodes/s per build and checks that the node counts agree.
#
# Example (MEMO, idle):
#   .\speed_ab.ps1 -A C:\astra\SelfPlay\SelfPlay-esimpA-perf.exe -B C:\astra\SelfPlay\SelfPlay-esimpB-perf.exe
param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [string]$Fens = "C:\Learn\Tune\hvar\speed\tests.fen",
    [int]$Depth = 10,
    [int]$Rounds = 6
)

$ErrorActionPreference = "Stop"
$work = Join-Path $env:TEMP "speed_ab"
New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $work
$res = @{ "A" = @(); "B" = @() }
$nodes = @{ "A" = @(); "B" = @() }
for ($r = 1; $r -le $Rounds; $r++) {
    $order = @("A", "B") | Get-Random -Count 2
    foreach ($v in $order) {
        $exe = if ($v -eq "A") { $A } else { $B }
        $out = & $exe -P -i $Fens -d $Depth 2>&1 | Out-String
        $nps = [int]([regex]::Match($out, 'Nodes/second:\s+(\d+)').Groups[1].Value)
        $tn = [long]([regex]::Match($out, 'Total nodes:\s+(\d+)').Groups[1].Value)
        $res[$v] += $nps
        $nodes[$v] += $tn
        Write-Host "round $r $v : $nps nodes/s ($tn nodes)"
    }
}
foreach ($v in @("A", "B")) {
    $m = ($res[$v] | Measure-Object -Average).Average
    $sd = [math]::Sqrt((($res[$v] | ForEach-Object { ($_ - $m) * ($_ - $m) }) | Measure-Object -Sum).Sum / ($Rounds - 1))
    Write-Host ("{0}: mean {1:N0} nodes/s, sd {2:N0}" -f $v, $m, $sd)
}
$ma = ($res["A"] | Measure-Object -Average).Average
$mb = ($res["B"] | Measure-Object -Average).Average
Write-Host ("A against B: {0:+0.0;-0.0} %" -f (100 * ($ma - $mb) / $mb))
if (($nodes["A"] | Select-Object -Unique).Count -ne 1 -or $nodes["A"][0] -ne $nodes["B"][0]) {
    Write-Host "WARNING: node counts differ, the builds do not search the same tree"
}
