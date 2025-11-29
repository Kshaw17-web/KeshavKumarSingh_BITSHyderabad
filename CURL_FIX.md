# Fix for curl "Failed to open/read local data" Error in PowerShell

## Problem
PowerShell's `curl.exe` (which is an alias for `Invoke-WebRequest`) has different syntax than Unix curl.

## Solutions

### Option 1: Use PowerShell Native Command (Recommended)
```powershell
# Use Invoke-RestMethod instead of curl
$uri = "http://127.0.0.1:8000/api/v1/hackrx/run"
$form = @{
    document = Get-Item -Path "C:\temp\train_sample_2.pdf"
}
$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
$response | ConvertTo-Json -Depth 10
```

### Option 2: Use Test Script
```powershell
.\test_upload.ps1 "C:\temp\train_sample_2.pdf"
```

### Option 3: Fix curl.exe Syntax
If you want to use curl.exe, try:
```powershell
# Use full path with quotes and proper escaping
curl.exe -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" -F "document=@C:\temp\train_sample_2.pdf"
```

### Option 4: Use Python Script
```bash
python TEST_API_UPLOAD.py "C:\temp\train_sample_2.pdf"
```

## Common Issues

1. **File path with spaces**: Use quotes around the path
2. **File doesn't exist**: Check with `Test-Path "C:\temp\train_sample_2.pdf"`
3. **PowerShell alias**: `curl` in PowerShell is an alias for `Invoke-WebRequest`, use `curl.exe` for actual curl

## Recommended Approach

Use the PowerShell script `test_upload.ps1` for reliable file uploads in PowerShell.

