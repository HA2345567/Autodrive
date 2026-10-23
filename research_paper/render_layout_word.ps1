param([string]$InputDocument,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$renderWord=New-Object -ComObject Word.Application
$renderWord.Visible=$false
$renderWord.DisplayAlerts=0
$renderDoc=$renderWord.Documents.Open($InputDocument,$false,$true)
Write-Output 'Opened document'
$renderDoc.ExportAsFixedFormat($OutputPdf,17)
Write-Output 'Exported PDF'
$renderDoc.Close(0)
$renderWord.Quit()
Write-Output 'Closed render instance'
