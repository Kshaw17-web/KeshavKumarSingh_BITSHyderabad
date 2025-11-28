@echo off
setlocal enabledelayedexpansion
cd /d C:\datathon_work

if not exist results (
    mkdir results
)

echo [run_tests] launching FastAPI test server via Uvicorn...
powershell -NoProfile -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "$server = Start-Process -FilePath 'python' -ArgumentList '-m','uvicorn','src.api:app','--host','127.0.0.1','--port','8000' -PassThru;" ^
    "Start-Sleep -Seconds 5;" ^
    "$docs = @('C:\\datathon_work\\data\\raw\\Sample_Document 1.pdf','C:\\datathon_work\\data\\raw\\Sample_Document 3.pdf','C:\\datathon_work\\data\\raw\\Sample_Document 1.pdf');" ^
    "for ($i = 0; $i -lt 3; $i++) { $body = @{ document = $docs[$i] } | ConvertTo-Json; try { $resp = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/extract-bill-data' -Body $body -ContentType 'application/json'; $resp | ConvertTo-Json -Depth 5 | Set-Content -Path ('C:\\datathon_work\\results\\response_' + ($i+1) + '.json'); Add-Content -Path 'C:\\datathon_work\\results\\smoke_test_outputs.txt' -Value ('Request ' + ($i+1) + ':' + [Environment]::NewLine + ($resp | ConvertTo-Json -Depth 5) + [Environment]::NewLine); } catch { Add-Content -Path 'C:\\datathon_work\\results\\smoke_test_outputs.txt' -Value ('Request ' + ($i+1) + ' FAILED: ' + $_.Exception.Message); } }" ^
    "Stop-Process -Id $server.Id"

echo [run_tests] complete. Review results\ for response artifacts.
endlocal

