$ErrorActionPreference = 'Stop'
$inputDocument = 'D:\Documents\AutoDrive\output\docx\AutoDrive_Research_Paper_Latest.docx'
$renderFolder = 'D:\Documents\AutoDrive\tmp\docx_latest\render'
New-Item -ItemType Directory -Path $renderFolder -Force | Out-Null
$pdfOutput = Join-Path $renderFolder 'AutoDrive_Research_Paper_Latest.pdf'
$wordApp = $null
$wordDocument = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordDocument = $wordApp.Documents.Open($inputDocument, $false, $true)
    Write-Output 'Document opened for rendering'
    $wordDocument.ExportAsFixedFormat($pdfOutput, 17)
    Write-Output $pdfOutput
}
finally {
    if ($null -ne $wordDocument) { $wordDocument.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordDocument) }
    if ($null -ne $wordApp) { $wordApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) }
}
