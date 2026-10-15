# Gauntlet of Barbarossa against other engines with fastchess, for training data from the games
# (Barbarossa option -d: the search leaf of every move, joined with the PGN results by uci_data.py).
#
# Creates <Root>\<Name>\ with:
#   eng\        working directory of Barbarossa: one .data file per process (normal log off: -l 5)
#   games.pgn   the games (results for the join)
#   fastchess.log, command.txt (the full fastchess command), openings.txt (the opening range used)
# The openings are taken sequentially; <Root>\next-opening.txt remembers where the next run starts, so that
# runs do not repeat openings. Afterwards:
#   python uci_data.py --data <Root>\<Name>\eng --pgn <Root>\<Name>\games.pgn --engine Barbarossa
#       --out <data dir> --name <Name> --shards 4
#
# Example (MEMO):
#   .\run_uci_games.ps1 -Name ug-test1 -Games 100
#   .\run_uci_games.ps1 -Name ug-a -Games 4800 -Opponents Tantabus,Luna
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][int]$Games,      # total games (all opponents), rounded up to whole pairs
    # Barbarossa scored 56-62% against ECE-X2, Admete, Odonata (ug-check), 46-60% against Tantabus, Luna,
    # Weiawaga (ug-cand1); ECE12 (15% for it in ug-test1) and CT800 (32%) are too weak
    [string[]]$Opponents = @("ECE-X2", "Admete", "Odonata", "Tantabus", "Luna", "Weiawaga"),
    [string]$Tc = "10+0.1",
    [int]$Concurrency = 6,
    [string]$Barbarossa = "C:\Engines\Barbarossa\Barbarossa-0.8.0-ucidat.exe",   # needs option -d
    [string]$Config = "",                           # config file for Barbarossa (-c), e.g. eval parameters
    [string]$EnginesDir = "C:\Engines",
    [string]$Openings = "C:\Learn\Tune\hvar\open\open-moves-uniq-fm1.fen",
    [int]$Start = 0,                                # first opening (1-based); 0 = from next-opening.txt
    [string]$Root = "C:\Learn\Tune\hvar\ugames",
    [string]$FastChess = "C:\Program Files (x86)\fastchess-windows-x86-64\fastchess.exe"
)

$ErrorActionPreference = "Stop"

$known = @{
    "ECE12"   = "ECE\ECE12.exe"
    "ECE-X2"  = "ECE\ECE-X2\ECE-X2.exe"
    "CT800"   = "ct800-v1.43\binaries\Windows\CT800_V1.43_x64.exe"
    "Admete"  = "Other\admete_1.5.0_windows_x86-64.exe"
    "Odonata" = "Other\odonata-0.5.1-windows-modern.exe"
    "Weiawaga" = "Other\Weiawaga-v4.0-x86_64-pc-windows-msvc.exe"
    "Tantabus" = "Other\tantabus-windows-2019-x86-64-v3.exe"
    "Luna"    = "Other\luna_2.0.0_avx2_win_x86-64.exe"
}
# UCI options per engine (checked with "uci"; all search with 1 thread by default): 16 MB hash like
# Barbarossa (its minimum), threads 1 explicitly, own books off (games start from our openings)
$engineOptions = @{
    "ECE12"   = @()
    "ECE-X2"  = @("option.ECE-X2 Threads=1", "option.ECE-X2 Hash Size=16", "option.ECE-X2 Use Book=false")
    "CT800"   = @("option.Hash=16", "option.OwnBook=false")
    "Admete"  = @("option.Hash=16")
    "Odonata" = @("option.Threads=1", "option.Hash=16")
    "Weiawaga" = @("option.Threads=1", "option.Hash=16")
    "Tantabus" = @("option.Hash=16")
    "Luna"    = @("option.Hash=16")
}

$dir = Join-Path $Root $Name
if (Test-Path $dir) { throw "$dir exists already: choose another name" }
foreach ($f in @($Barbarossa, $Openings, $FastChess)) {
    if (-not (Test-Path $f)) { throw "not found: $f" }
}

$engineArgs = @()
foreach ($o in $Opponents) {
    if (-not $known.ContainsKey($o)) { throw "unknown opponent $o (known: $($known.Keys -join ', '))" }
    $exe = Join-Path $EnginesDir $known[$o]
    if (-not (Test-Path $exe)) { throw "not found: $exe" }
    $engineArgs += @("-engine", "cmd=$exe", "name=$o", "dir=$(Split-Path $exe)") + $engineOptions[$o]
}

$rounds = [int][math]::Ceiling($Games / (2.0 * $Opponents.Count))
$nextFile = Join-Path $Root "next-opening.txt"
New-Item -ItemType Directory -Force $Root | Out-Null
if ($Start -le 0) {
    $Start = if (Test-Path $nextFile) { [int](Get-Content $nextFile) } else { 1 }
}
$used = $rounds                                     # gauntlet: one opening per round, for all opponents
Set-Content -Path $nextFile -Value ($Start + $used)

$eng = Join-Path $dir "eng"
New-Item -ItemType Directory -Force $eng | Out-Null
$pgn = Join-Path $dir "games.pgn"

$barArgs = "-l 5 -d"
if ($Config) {
    if (-not (Test-Path $Config)) { throw "not found: $Config" }
    $barArgs += " -c $Config"
}
$fcArgs = @("-engine", "cmd=$Barbarossa", "name=Barbarossa", "args=$barArgs", "dir=$eng") + $engineArgs + @(
    "-each", "tc=$Tc", "proto=uci",
    "-tournament", "gauntlet", "-seeds", "1",
    "-rounds", "$rounds", "-games", "2", "-repeat",
    "-concurrency", "$Concurrency",
    "-openings", "file=$Openings", "format=epd", "order=sequential", "start=$Start",
    "-pgnout", "file=$pgn",
    "-log", "file=$(Join-Path $dir 'fastchess.log')", "level=warn",
    "-ratinginterval", "50",
    "-recover"
)

Set-Content -Path (Join-Path $dir "command.txt") -Value ("`"$FastChess`" " + ($fcArgs -join " "))
Set-Content -Path (Join-Path $dir "openings.txt") -Value "$Openings start $Start, $used openings"
Write-Host "$Name : $($rounds * 2 * $Opponents.Count) games ($rounds rounds x 2 x $($Opponents.Count) opponents), tc $Tc,"
Write-Host "  concurrency $Concurrency, openings from $Start, output $dir"

& $FastChess @fcArgs
