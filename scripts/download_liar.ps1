mkdir data -ErrorAction SilentlyContinue | Out-Null
Invoke-WebRequest "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip" -OutFile "data\liar_dataset.zip"
Expand-Archive "data\liar_dataset.zip" -DestinationPath "data" -Force
Write-Host "LIAR downloaded to data/"
