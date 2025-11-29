# PowerShell script to test file upload to API
# Usage: .\test_upload.ps1 "C:\temp\train_sample_2.pdf"

param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

# Check if file exists
if (-not (Test-Path $FilePath)) {
    Write-Host "ERROR: File not found: $FilePath" -ForegroundColor Red
    exit 1
}

Write-Host "Uploading: $FilePath" -ForegroundColor Green

# Use Invoke-RestMethod (PowerShell native) instead of curl
try {
    $uri = "http://127.0.0.1:8000/api/v1/hackrx/run"
    
    # Create multipart form data
    $form = @{
        document = Get-Item -Path $FilePath
    }
    
    $response = Invoke-RestMethod -Uri $uri -Method Post -Form $form -ContentType "multipart/form-data"
    
    Write-Host "`n=== RESPONSE ===" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10
    
    if ($response.is_success) {
        Write-Host "`n✓ SUCCESS!" -ForegroundColor Green
        $itemCount = $response.data.total_item_count
        $amount = $response.data.reconciled_amount
        Write-Host "Items extracted: $itemCount" -ForegroundColor Green
        Write-Host "Total amount: $amount" -ForegroundColor Green
    } else {
        Write-Host "`n✗ FAILED" -ForegroundColor Red
        Write-Host "Error: $($response.message)" -ForegroundColor Red
        if ($response.traceback) {
            Write-Host "`nTraceback (last 10 lines):" -ForegroundColor Yellow
            $response.traceback -split "`n" | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
        }
    }
    
} catch {
    Write-Host "`nERROR: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails) {
        Write-Host "Details: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
    }
    exit 1
}

