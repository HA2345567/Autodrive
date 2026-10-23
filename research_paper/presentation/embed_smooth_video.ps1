$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\Documents\AutoDrive'
$sourceDeck = Join-Path $taskRoot 'output\presentation\AutoDrive_Final_Presentation_Ready.pptx'
$buildFolder = Join-Path $taskRoot 'tmp\presentation_video'
$draftDeck = Join-Path $buildFolder 'candidate-with-video.pptx'
$videoPath = Join-Path $taskRoot 'output\video\AutoDrive_20s_Demo_Smooth_Steering.mp4'
$posterPath = Join-Path $taskRoot 'output\video\AutoDrive_20s_Demo_Smooth_Steering_encoded_0000.png'
New-Item -ItemType Directory -Path $buildFolder -Force | Out-Null
Copy-Item -LiteralPath $sourceDeck -Destination $draftDeck
$powerPoint = $null
$deck = $null
$initialDeckCount = 0
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $initialDeckCount = $powerPoint.Presentations.Count
    $deck = $powerPoint.Presentations.Open($draftDeck, 0, 0, 0)
    $slide = $deck.Slides.Item(19)
    $shapeReport = @()
    for ($i = 1; $i -le $slide.Shapes.Count; $i++) {
        $shape = $slide.Shapes.Item($i)
        $value = if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) { $shape.TextFrame.TextRange.Text } else { '' }
        $shapeReport += [ordered]@{id=$shape.Id;name=$shape.Name;type=[int]$shape.Type;left=$shape.Left;top=$shape.Top;width=$shape.Width;height=$shape.Height;text=$value}
    }
    $shapeReport | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $buildFolder 'source-slide-19.json') -Encoding UTF8
    $slide.Export((Join-Path $buildFolder 'before-slide-19.png'), 'PNG', 1920, 1080)
    # Replace the existing prototype screenshot while retaining title and footer.
    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $slide.Shapes.Item($i)
        if ([int]$shape.Type -eq 13 -or [int]$shape.Type -eq 11) {
            $shape.Delete()
        } elseif ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
            $value = $shape.TextFrame.TextRange.Text
            if ($value -eq 'Prototype visualization') {
                $shape.TextFrame.TextRange.Text = 'AutoDrive demonstration'
            } elseif ($value.StartsWith('Display demonstration only.')) {
                $shape.TextFrame.TextRange.Text = '20-second offline demo with smooth AI steering'
                $shape.Left = 54
                $shape.Top = 488.25
                $shape.Width = 852
                $shape.Height = 27
            }
        }
    }
    # LinkToFile=false and SaveWithDocument=true embed the complete MP4.
    $movie = $slide.Shapes.AddMediaObject2($videoPath, 0, -1, 144, 105.75, 672, 378)
    $movie.Name = 'AutoDrive_20s_Smooth_Steering_Demo'
    $movie.AlternativeText = '20-second AutoDrive offline demonstration with lane and object predictions and a smoothly animated AI steering wheel'
    $movie.MediaFormat.SetDisplayPictureFromFile($posterPath)
    $play = $movie.AnimationSettings.PlaySettings
    $play.PlayOnEntry = -1
    $play.PauseAnimation = 0
    $play.HideWhileNotPlaying = 0
    $play.LoopUntilStopped = 0
    $play.RewindMovie = -1
    $notes = @'
Ab yeh 20-second recorded demo dekhiye. Road view par lane masks aur object boxes dikh rahe hain. Right side ka wheel CNN plus Huber AI prediction ko follow karta hai, aur movement par original cockpit smoothing lagi hai. Numeric AI angle raw prediction hai. Recorded angle aur error comparison ke liye dikhte hain. Yeh offline rendered playback 20 FPS par hai. Source frames 2400-2799 steering ke original training partition mein hain, isliye ise held-out evaluation ya live inference speed ka result nahi kahenge.

Source: output/video/AutoDrive_20s_Demo_Smooth_Steering.mp4 and its prediction cache, manifest and per-frame CSV. Video duration 20 seconds, 400 source frames, 1280 by 720 pixels, 20 FPS. The presentation embeds the MP4 and starts playback on entering this slide.
'@
    foreach ($placeholder in $slide.NotesPage.Shapes.Placeholders) {
        if ([int]$placeholder.PlaceholderFormat.Type -eq 2) {
            $placeholder.TextFrame.TextRange.Text = $notes
        }
    }
    $deck.Save()
    $slide.Export((Join-Path $buildFolder 'after-slide-19.png'), 'PNG', 1920, 1080)
    [ordered]@{candidate=$draftDeck;slide_count=$deck.Slides.Count;video_slide=19;media_name=$movie.Name;media_type=[int]$movie.MediaType;video_duration_ms=$movie.MediaFormat.Length;play_on_entry=[int]$play.PlayOnEntry;slide_width=$deck.PageSetup.SlideWidth;slide_height=$deck.PageSetup.SlideHeight} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $buildFolder 'embedding-result.json') -Encoding UTF8
    Get-Content -LiteralPath (Join-Path $buildFolder 'embedding-result.json')
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if ($null -ne $powerPoint -and $initialDeckCount -eq 0) { $powerPoint.Quit() }
    if ($null -ne $deck) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck) }
    if ($null -ne $powerPoint) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) }
}
