# Timed match with fastchess: a candidate (default qthr, quiet sort with escapes first; hpqt = qthr + history
# prior; rfut = reverse futility pruning) against the parent dev090, fast builds Barbarossa-<name>-fast.exe (flag -qscollect); output (PGN,
# log) in C:\Learn\Tune\hvar\fast\<Name>
param(
    [string]$Name = "qthr-dev090-tc10",
    [string]$Cand = "qthr",
    [string]$Opp = "dev090",
    [string]$CandName = "",     # candidate name in the games, default its build; needed when Cand = Opp
    [string]$CandParams = "",   # candidate parameters (engine option -p), e.g. epLmrFix=14
    [string]$Tc = "10+0.1",
    [int]$Rounds = 2000,
    [int]$Concurrency = 8,
    [int]$Start = 1,
    [switch]$Sprt,              # SPRT (logistic Elo) with the bounds below, stops early on a decision
    [double]$Elo0 = 0,
    [double]$Elo1 = 5,
    [string]$FastChess = "C:\Program Files (x86)\fastchess-windows-x86-64\fastchess.exe"
)
$base = "C:\Learn\Tune\hvar\fast"
$dir = Join-Path $base $Name
$eng = Join-Path $dir "eng"
New-Item -ItemType Directory -Force $eng | Out-Null
if (-not $CandName) { $CandName = $Cand }
$candArgs = if ($CandParams) { "-l 5 -p $CandParams" } else { "-l 5" }
$fcArgs = @(
    "-engine", "cmd=$base\Barbarossa-$Cand-fast.exe", "name=$CandName", "args=$candArgs", "dir=$eng",
    "-engine", "cmd=$base\Barbarossa-$Opp-fast.exe", "name=$Opp", "args=-l 5", "dir=$eng",
    "-each", "tc=$Tc", "proto=uci",
    "-rounds", "$Rounds", "-games", "2", "-repeat",
    "-concurrency", "$Concurrency",
    "-openings", "file=C:\Learn\Tune\hvar\open\open-moves-uniq-fm1.fen", "format=epd", "order=random", "start=$Start",
    "-pgnout", "file=$(Join-Path $dir 'games.pgn')",
    "-log", "file=$(Join-Path $dir 'fastchess.log')", "level=warn",
    "-ratinginterval", "100",
    "-recover"
)
if ($Sprt) {
    $fcArgs += @("-sprt", "elo0=$Elo0", "elo1=$Elo1", "alpha=0.05", "beta=0.05", "model=logistic")
}
Set-Content -Path (Join-Path $dir "command.txt") -Value ("`"$FastChess`" " + ($fcArgs -join " "))
# fastchess sometimes deadlocks after the SPRT decision (no CPU, engines left, no reaction to Ctrl-C): the games and
# the result are complete by then, so a separate process ends it and its engines if it is still there after the grace time
$graceSec = 60
$script:finished = $false
$script:decided = $false
& $FastChess @fcArgs | ForEach-Object {
    $_
    if ($_ -match '^Finished match') { $script:finished = $true }
    if ($_ -match '^SPRT .* completed') {
        $fc = (Get-CimInstance Win32_Process -Filter "ParentProcessId=$PID AND Name='$(Split-Path $FastChess -Leaf)'").ProcessId
        $script:decided = $true
        $kill = "Start-Sleep $graceSec; if (Get-Process -Id $fc -ErrorAction SilentlyContinue | Where-Object ProcessName -eq 'fastchess') { " +
                "Get-CimInstance Win32_Process -Filter 'ParentProcessId=$fc' | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }; " +
                "Stop-Process -Id $fc -Force }"
        Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", $kill
    }
}
if ($script:decided -and -not $script:finished) { Write-Host "fastchess did not end by itself and was stopped; the result above is complete" }
