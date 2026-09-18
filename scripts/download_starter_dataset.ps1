param(
  [string]$Destination = 'E:\ForestFire\starter_dataset',
  [int]$Count = 30
)

$ErrorActionPreference = 'Stop'
$mtbsLayer = 'https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/63/query'
$stacSearch = 'https://earth-search.aws.element84.com/v1/search'

function Save-Json([object]$Object, [string]$Path) {
  $Object | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Get-Scene([double]$Lon, [double]$Lat, [datetime]$From, [datetime]$To) {
  $bbox = '{0},{1},{2},{3}' -f ($Lon - 0.02), ($Lat - 0.02), ($Lon + 0.02), ($Lat + 0.02)
  $datetime = '{0:yyyy-MM-dd}T00:00:00Z/{1:yyyy-MM-dd}T23:59:59Z' -f $From, $To
  $query = "?collections=sentinel-2-l2a&bbox=$bbox&datetime=$([uri]::EscapeDataString($datetime))&limit=100"
  $result = Invoke-RestMethod -Uri ($stacSearch + $query)
  return @($result.features | Where-Object { $_.properties.'eo:cloud_cover' -le 20 } |
    Sort-Object { $_.properties.'eo:cloud_cover' } |
    Select-Object -First 1)[0]
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$labelsDir = Join-Path $Destination 'labels'
$firesDir = Join-Path $Destination 'fires'
New-Item -ItemType Directory -Force -Path $labelsDir, $firesDir | Out-Null

$where = [uri]::EscapeDataString("year = 2020 AND fire_type = 'Wildfire'")
$uri = "$mtbsLayer?where=$where&outFields=*&returnGeometry=true&orderByFields=acres%20DESC&resultRecordCount=$Count&f=geojson"
$events = Invoke-RestMethod -Uri $uri
Save-Json $events (Join-Path $labelsDir 'mtbs_boundaries_2020.geojson')

$manifest = [System.Collections.Generic.List[object]]::new()
foreach ($feature in $events.features) {
  $p = $feature.properties
  $fireId = $p.fire_id
  $eventDate = [datetime]::new(2020, [int]$p.startmonth, [int]$p.startday)
  $fireDir = Join-Path $firesDir $fireId
  New-Item -ItemType Directory -Force -Path $fireDir | Out-Null
  try {
    $pre = Get-Scene -Lon ([double]$p.longitude) -Lat ([double]$p.latitude) -From $eventDate.AddDays(-45) -To $eventDate.AddDays(-5)
    $post = Get-Scene -Lon ([double]$p.longitude) -Lat ([double]$p.latitude) -From $eventDate.AddDays(10) -To $eventDate.AddDays(60)
    if ($null -eq $pre -or $null -eq $post) { throw 'No low-cloud Sentinel-2 scene found in the target windows.' }

    Save-Json $pre (Join-Path $fireDir 'pre_item.json')
    Save-Json $post (Join-Path $fireDir 'post_item.json')
    foreach ($scene in @(@{Name='pre'; Item=$pre}, @{Name='post'; Item=$post})) {
      foreach ($asset in @(@{Name='B04'; Key='red'}, @{Name='B08'; Key='nir'}, @{Name='B12'; Key='swir22'})) {
        $outFile = Join-Path $fireDir ("{0}_{1}.tif" -f $scene.Name, $asset.Name)
        Invoke-WebRequest -Uri $scene.Item.assets.($asset.Key).href -OutFile $outFile
      }
    }
    $status = 'downloaded'
    Write-Host "Downloaded $fireId"
  } catch {
    $status = "failed: $($_.Exception.Message)"
    Write-Warning "$fireId - $status"
  }
  $manifest.Add([pscustomobject]@{
    fire_id=$fireId; fire_name=$p.fire_name; event_date=$eventDate.ToString('yyyy-MM-dd'); acres=$p.acres;
    longitude=$p.longitude; latitude=$p.latitude; status=$status
  })
}
$manifest | Export-Csv -LiteralPath (Join-Path $Destination 'manifest.csv') -NoTypeInformation -Encoding utf8
@"
# Starter wildfire dataset

Contains the 30 largest 2020 MTBS wildfires, their official burned-area boundaries,
and up to six Sentinel-2 L2A COG assets per fire: B04, B08, B12 before and after the event.

Files are raw, full Sentinel tiles. Crop and align the scenes to each boundary before training.
The manifest.csv status column records incomplete cases for retry.
"@ | Set-Content -LiteralPath (Join-Path $Destination 'README.md') -Encoding utf8
