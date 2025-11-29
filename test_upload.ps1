# PowerShell script to test file upload to API
param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

if (-not (Test-Path $FilePath)) {
    Write-Host "ERROR: File not found: $FilePath" -ForegroundColor Red
    exit 1
}

Write-Host "Uploading: $FilePath" -ForegroundColor Green

try {
    $uri = "http://127.0.0.1:8000/api/v1/hackrx/run"
    
    # Use Invoke-WebRequest for file upload (works in all PowerShell versions)
    $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)
    $fileName = [System.IO.Path]::GetFileName($FilePath)
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"document`"; filename=`"$fileName`"",
        "Content-Type: application/pdf",
        "",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
        "--$boundary--"
    ) -join $LF
    
    $bodyBytes = [System.Text.Encoding]::GetEncoding("iso-8859-1").GetBytes($bodyLines)
    
    $response = Invoke-WebRequest -Uri $uri -Method Post -Body $bodyBytes -ContentType "multipart/form-data; boundary=$boundary"
    
    $jsonResponse = $response.Content | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "=== RESPONSE ===" -ForegroundColor Cyan
    $jsonResponse | ConvertTo-Json -Depth 10
    
    if ($jsonResponse.is_success) {
        Write-Host ""
        Write-Host "SUCCESS!" -ForegroundColor Green
        $itemCount = $jsonResponse.data.total_item_count
        $amount = $jsonResponse.data.reconciled_amount
        Write-Host "Items extracted: $itemCount" -ForegroundColor Green
        Write-Host "Total amount: $amount" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "FAILED" -ForegroundColor Red
        Write-Host "Error: $($jsonResponse.message)" -ForegroundColor Red
        Write-Host "Error field: $($jsonResponse.error)" -ForegroundColor Red
        if ($jsonResponse.traceback) {
            Write-Host ""
            Write-Host "Traceback (last 20 lines):" -ForegroundColor Yellow
            $lines = $jsonResponse.traceback -split "`n"
            $lines | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
        }
        if ($jsonResponse.traceback_file) {
            Write-Host ""
            Write-Host "Full traceback saved to: $($jsonResponse.traceback_file)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails) {
        Write-Host "Details:" -ForegroundColor Yellow
        Write-Host $_.ErrorDetails.Message -ForegroundColor Yellow
    }
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor Yellow
    }
    exit 1
}
