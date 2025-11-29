# Quick Upload Guide - PowerShell

## Problem
`curl.exe -F 'document=@C:\temp\train_sample_2.pdf'` fails in PowerShell with "Failed to open/read local data"

## Solutions

### ✅ Solution 1: Use PowerShell Native (Recommended)
```powershell
$uri = "http://127.0.0.1:8000/api/v1/hackrx/run"
$filePath = "C:\temp\train_sample_2.pdf"  # Change to your actual file path

# Check if file exists first
if (Test-Path $filePath) {
    $form = @{
        document = Get-Item -Path $filePath
    }
    $response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
    $response | ConvertTo-Json -Depth 10
} else {
    Write-Host "File not found: $filePath" -ForegroundColor Red
}
```

### ✅ Solution 2: Use Test Script
```powershell
.\test_upload.ps1 "C:\path\to\your\file.pdf"
```

### ✅ Solution 3: Fix curl.exe Syntax
```powershell
# Use double quotes and proper escaping
curl.exe -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" -F "document=@C:\temp\train_sample_2.pdf"
```

### ✅ Solution 4: Use Python Script
```powershell
python TEST_API_UPLOAD.py "C:\path\to\your\file.pdf"
```

## Find Your PDF Files
```powershell
# Search for PDF files
Get-ChildItem -Path "C:\" -Filter "*.pdf" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 10 FullName

# Or check common locations
Get-ChildItem "C:\Users\$env:USERNAME\Downloads\*.pdf" -ErrorAction SilentlyContinue
Get-ChildItem "C:\temp\*.pdf" -ErrorAction SilentlyContinue
```

## Test with Sample File
If you have training samples in the project:
```powershell
# Find training samples
Get-ChildItem -Path "data\raw\training_samples" -Filter "*.pdf" -Recurse | Select-Object -First 1 | ForEach-Object {
    $filePath = $_.FullName
    Write-Host "Testing with: $filePath"
    .\test_upload.ps1 $filePath
}
```

