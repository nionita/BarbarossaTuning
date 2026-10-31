# Node scan: search every position of a fen file to a fixed depth with every config of a folder and record
# the nodes per position (perf test of a deterministic SelfPlay build, one process per position, so every
# search starts fresh). The positions are split into -Shards ranges searched in parallel; each shard goes
# through all configs. Output per config and shard: <Out>\<config>-s<i>.csv with lines "index,nodes"
# (appended per position: a restarted scan continues where it stopped). node_scan_report.py evaluates it.
# A search over -TimeLimit seconds is stopped and recorded as nodes -1; the position is then not searched with
# the later configs of the shard (also -1), so the limit is paid about once per position. The config -First
# (the reference) runs first: a stopped scan has it, and it is usually the most expensive one.
#
# Example (MEMO):
#   .\node_scan.ps1 -Exe C:\Learn\Tune\hvar\nodescan\SelfPlay-rfth-perf.exe -Fens C:\Learn\Tune\hvar\nodescan\roots-2000.fen `
#       -CfgDir C:\Learn\Tune\hvar\nodescan\cfg -Out C:\Learn\Tune\hvar\nodescan\out -Shards 8 -Depth 12

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Fens,
    [Parameter(Mandatory = $true)][string]$CfgDir,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Shards = 8,
    [int]$Depth = 12,
    [int]$Count = 0,    # only the first Count positions (0 = all)
    [int]$TimeLimit = 600,      # seconds per search (0 = no limit)
    [string]$First = "off"      # config searched first
)

$ErrorActionPreference = "Stop"
foreach ($p in @($Exe, $Fens, $CfgDir)) { if (-not (Test-Path -LiteralPath $p)) { throw "Not found: $p" } }
$fenList = @(Get-Content -LiteralPath $Fens | Where-Object { $_.Trim() })
if ($Count -gt 0 -and $Count -lt $fenList.Count) { $fenList = $fenList[0..($Count - 1)] }
$cfgs = @(Get-ChildItem -LiteralPath $CfgDir -Filter *.cfg | Sort-Object @{ Expression = { $_.BaseName -ne $First } }, Name | ForEach-Object { $_.FullName })
if ($cfgs.Count -eq 0) { throw "No *.cfg in $CfgDir" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$n = $fenList.Count
Write-Host "$n positions, depth $Depth, $($cfgs.Count) configs, $Shards shards -> $Out"

$worker = {
    param($Exe, $FenList, $From, $To, $Cfgs, $Depth, $Out, $Shard, $TimeLimit)
    $tmp = Join-Path $Out "tmp-s$Shard.fen"
    $waitMs = if ($TimeLimit -gt 0) { $TimeLimit * 1000 } else { -1 }
    $skip = @{}
    # the progress display of the main script reads the same files: retry when locked
    function AddLine($path, $line) {
        foreach ($try in 1..50) {
            try { [IO.File]::AppendAllText($path, "$line`r`n"); return } catch { Start-Sleep -Milliseconds 100 }
        }
        throw "Cannot write $path"
    }
    foreach ($cfg in $Cfgs) {
        $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
        $csv = Join-Path $Out "$name-s$Shard.csv"
        $done = @{}
        if (Test-Path -LiteralPath $csv) {
            Get-Content -LiteralPath $csv | ForEach-Object {
                $f = $_ -split ","
                $done[[int]$f[0]] = $true
                if ([int64]$f[1] -lt 0) { $skip[[int]$f[0]] = $true }
            }
        }
        for ($i = $From; $i -lt $To; $i++) {
            if ($done.ContainsKey($i)) { continue }
            if ($skip.ContainsKey($i)) { AddLine $csv "$i,-1"; continue }
            Set-Content -LiteralPath $tmp -Value $FenList[$i] -Encoding Ascii
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Exe
            $psi.Arguments = "-P -i `"$tmp`" -d $Depth -c `"$cfg`""
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $p = [System.Diagnostics.Process]::Start($psi)
            $res = $p.StandardOutput.ReadToEndAsync()
            if (-not $p.WaitForExit($waitMs)) {
                $p.Kill()
                $p.WaitForExit()
                $skip[$i] = $true
                AddLine $csv "$i,-1"
                continue
            }
            if ($res.Result -notmatch '(?m)^Total nodes:\s+(\d+)') { throw "No node count: $name, position $i" }
            AddLine $csv "$i,$($Matches[1])"
        }
    }
}

$jobs = @()
foreach ($s in 1..$Shards) {
    $from = [int][math]::Floor(($s - 1) * $n / $Shards)
    $to = [int][math]::Floor($s * $n / $Shards)
    $jobs += Start-Job -ScriptBlock $worker -ArgumentList $Exe, $fenList, $from, $to, $cfgs, $Depth, $Out, $s, $TimeLimit
    Write-Host "shard $s : positions $from .. $($to - 1)"
}

# progress: positions done per config, every minute
while (($jobs | Where-Object { $_.State -eq "Running" }).Count -gt 0) {
    Start-Sleep -Seconds 60
    $parts = foreach ($cfg in $cfgs) {
        $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
        $c = 0
        Get-ChildItem -LiteralPath $Out -Filter "$name-s*.csv" | ForEach-Object {
            try { $c += @(Get-Content -LiteralPath $_.FullName -ErrorAction Stop).Count } catch { }     # locked by a shard
        }
        "$name $c"
    }
    Write-Host ("{0}  {1}" -f (Get-Date -Format HH:mm), ($parts -join ", "))
}
$failed = @($jobs | Where-Object { $_.State -ne "Completed" })
foreach ($j in $failed) { Receive-Job $j -ErrorAction Continue }
$jobs | Remove-Job -Force
if ($failed.Count -gt 0) { throw "$($failed.Count) shard(s) failed" }

# totals per config (the detailed evaluation: node_scan_report.py)
foreach ($cfg in $cfgs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
    $tot = [int64]0
    $c = 0
    Get-ChildItem -LiteralPath $Out -Filter "$name-s*.csv" | ForEach-Object {
        Get-Content -LiteralPath $_.FullName | ForEach-Object {
            $v = [int64](($_ -split ",")[1])
            if ($v -gt 0) { $tot += $v; $c++ }
        }
    }
    Write-Host ("{0,-12} {1,5} positions  total nodes {2,13}" -f $name, $c, $tot)
}
