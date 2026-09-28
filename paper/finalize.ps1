# Update every SEQ/REF field, repaginate, report the page count and export a PDF.
# Word must do this: python-docx writes the field codes but cannot evaluate them.
param(
    [string]$Docx = "cross_vehicle_nvs_wevj_submission.docx",
    [switch]$NoPdf
)

$ErrorActionPreference = "Stop"
$full = (Resolve-Path $Docx).Path
$pdf  = [System.IO.Path]::ChangeExtension($full, ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full)

    # two passes: the first resolves REF targets, the second settles the page
    # numbers those references shift
    for ($i = 0; $i -lt 2; $i++) {
        $doc.Fields.Update() | Out-Null
        foreach ($story in $doc.StoryRanges) { $story.Fields.Update() | Out-Null }
    }
    $doc.Repaginate()

    $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
    $words = $doc.ComputeStatistics(0)   # wdStatisticWords

    # any field Word could not resolve shows as an error string
    $bad = 0
    foreach ($f in $doc.Fields) {
        if ($f.Result.Text -match "Error!") { $bad++ }
    }

    Write-Output ("pages: {0}   words: {1}   field errors: {2}" -f $pages, $words, $bad)

    $doc.Save()
    if (-not $NoPdf) {
        $doc.ExportAsFixedFormat($pdf, 17)   # wdExportFormatPDF
        Write-Output ("pdf: {0}" -f $pdf)
    }
    $doc.Close(0)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
