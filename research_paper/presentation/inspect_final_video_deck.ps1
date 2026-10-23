$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\Documents\AutoDrive'
$finalDeck = Join-Path $taskRoot 'output\presentation\AutoDrive_Final_Presentation_With_Smooth_Video.pptx'
$renderFolder = Join-Path $taskRoot 'tmp\presentation_video\final_renders'
New-Item -ItemType Directory -Path $renderFolder -Force | Out-Null
$powerPoint = $null
$deck = $null
$initialDeckCount = 0
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $initialDeckCount = $powerPoint.Presentations.Count
    $deck = $powerPoint.Presentations.Open($finalDeck, -1, 0, 0)
    $movie = $deck.Slides.Item(19).Shapes.Item('AutoDrive_20s_Smooth_Steering_Demo')
    $deck.Export($renderFolder, 'PNG', 1280, 720)
    [ordered]@{final=$finalDeck;slides=$deck.Slides.Count;video_slide=19;duration_ms=$movie.MediaFormat.Length;play_on_entry=[int]$movie.AnimationSettings.PlaySettings.PlayOnEntry;rendered_slides=(Get-ChildItem -LiteralPath $renderFolder -Filter '*.PNG').Count;opened_in_powerpoint=$true} | ConvertTo-Json
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if ($null -ne $powerPoint -and $initialDeckCount -eq 0) { $powerPoint.Quit() }
    if ($null -ne $deck) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck) }
    if ($null -ne $powerPoint) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) }
}
