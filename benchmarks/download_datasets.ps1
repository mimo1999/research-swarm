param(
    [string]$OutputDir = "data/benchmarks"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Download-File {
    param(
        [string]$Url,
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Already downloaded: $Destination"
        return
    }

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partial = "$Destination.partial"

    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
    Move-Item -LiteralPath $partial -Destination $Destination
}

function Expand-ZipOnce {
    param(
        [string]$Archive,
        [string]$Destination,
        [string]$ExpectedPath
    )

    if (Test-Path -LiteralPath $ExpectedPath) {
        Write-Host "Already extracted: $ExpectedPath"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}

$root = [System.IO.Path]::GetFullPath((Join-Path $PWD $OutputDir))
$archives = Join-Path $root "_archives"
New-Item -ItemType Directory -Force -Path $archives | Out-Null

# ALCE: ASQA, QAMPARI, and ELI5 with the authors' precomputed retrieval results.
$alceArchive = Join-Path $archives "ALCE-data.tar"
Download-File `
    "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar" `
    $alceArchive
$alceDir = Join-Path $root "alce"
if (-not (Test-Path -LiteralPath $alceDir)) {
    New-Item -ItemType Directory -Force -Path $alceDir | Out-Null
    tar -xf $alceArchive -C $alceDir
}

# HotpotQA distractor setting: the benchmark split suited to retrieval experiments.
$hotpotDir = Join-Path $root "hotpotqa"
Download-File `
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/train/0000.parquet" `
    (Join-Path $hotpotDir "train-0000.parquet")
Download-File `
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/train/0001.parquet" `
    (Join-Path $hotpotDir "train-0001.parquet")
Download-File `
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet" `
    (Join-Path $hotpotDir "validation-0000.parquet")

# SciFact's official release contains claims, evidence rationales, and abstracts.
$scifactArchive = Join-Path $archives "scifact-data.tar.gz"
Download-File `
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz" `
    $scifactArchive
$scifactDir = Join-Path $root "scifact"
if (-not (Test-Path -LiteralPath $scifactDir)) {
    New-Item -ItemType Directory -Force -Path $scifactDir | Out-Null
    tar -xzf $scifactArchive -C $scifactDir
}

# BEIR subsets selected for scientific, medical, and argument retrieval.
$beirRoot = Join-Path $root "beir"
foreach ($dataset in @("scifact", "nfcorpus", "arguana")) {
    $archive = Join-Path $archives "beir-$dataset.zip"
    Download-File `
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$dataset.zip" `
        $archive
    Expand-ZipOnce $archive $beirRoot (Join-Path $beirRoot $dataset)
}

$manifest = Join-Path $root "SHA256SUMS.txt"
$downloadedFiles = @(
    Get-ChildItem -LiteralPath $archives -File
    Get-ChildItem -LiteralPath $hotpotDir -File
)
$downloadedFiles |
    Sort-Object Name |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        $relativePath = $_.FullName.Substring($root.Length).TrimStart("\").Replace("\", "/")
        "$($hash.Hash.ToLowerInvariant())  $relativePath"
    } |
    Set-Content -LiteralPath $manifest -Encoding ascii

Write-Host "Datasets are ready under $root"
Write-Host "Archive checksums written to $manifest"
