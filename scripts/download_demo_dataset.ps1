param(
  [string]$Destination = 'E:\ForestFire\demo_dataset'
)

$ErrorActionPreference = 'Stop'
$fireId = 'CA3720111927220200905' # Creek Fire, California, 2020
$mtbsLayer = 'https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/63/query'
$stacSearch = 'https://earth-search.aws.element84.com/v1/search'

function Save-Json([object]$Object, [string]$Path) {
  $Object | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Download-File([string]$Url, [string]$Path) {
  & curl.exe --fail --location --retry 3 --continue-at - --output $Path $Url
  if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
}

function Get-Scene([double]$Lon, [double]$Lat, [datetime]$From, [datetime]$To) {
  $culture = [Globalization.CultureInfo]::InvariantCulture
  $coords = @(($Lon - 0.02), ($Lat - 0.02), ($Lon + 0.02), ($Lat + 0.02)) | ForEach-Object { $_.ToString($culture) }
  $bbox = [string]::Join(',', [string[]]$coords)
  $datetime = '{0:yyyy-MM-dd}T00:00:00Z/{1:yyyy-MM-dd}T23:59:59Z' -f $From, $To
  $uri = "$($stacSearch)?collections=sentinel-2-l2a&bbox=$bbox&datetime=$([uri]::EscapeDataString($datetime))&limit=100"
  $result = Invoke-RestMethod -Uri $uri
  return @($result.features | Where-Object { $_.properties.'eo:cloud_cover' -le 5 } |
    Sort-Object { $_.properties.'eo:cloud_cover' } | Select-Object -First 1)[0]
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$where = [uri]::EscapeDataString("fire_id = '$fireId'")
$event = Invoke-RestMethod -Uri "$($mtbsLayer)?where=$where&outFields=*&returnGeometry=true&f=geojson"
if ($event.features.Count -ne 1) { throw "MTBS boundary for $fireId was not found." }
$feature = $event.features[0]
$p = $feature.properties
$eventDate = [datetime]::new([int]$p.year, [int]$p.startmonth, [int]$p.startday)
$pre = Get-Scene -Lon ([double]$p.longitude) -Lat ([double]$p.latitude) -From $eventDate.AddDays(-20) -To $eventDate.AddDays(-5)
$post = Get-Scene -Lon ([double]$p.longitude) -Lat ([double]$p.latitude) -From $eventDate.AddDays(20) -To $eventDate.AddDays(50)
if ($null -eq $pre -or $null -eq $post) { throw 'No suitable low-cloud Sentinel-2 pair found.' }

Save-Json $event (Join-Path $Destination 'burn_boundary.geojson')
Save-Json $pre (Join-Path $Destination 'pre_item.json')
Save-Json $post (Join-Path $Destination 'post_item.json')
foreach ($scene in @(@{Name='pre'; Item=$pre}, @{Name='post'; Item=$post})) {
  foreach ($asset in @(@{Name='B08'; Key='nir'}, @{Name='B12'; Key='swir22'})) {
    Download-File -Url $scene.Item.assets.($asset.Key).href -Path (Join-Path $Destination ("{0}_{1}.tif" -f $scene.Name, $asset.Name))
  }
}
[pscustomobject]@{
  fire_id=$p.fire_id; fire_name=$p.fire_name; event_date=$eventDate.ToString('yyyy-MM-dd'); acres=$p.acres;
  pre_scene=$pre.id; post_scene=$post.id; method='dNBR baseline; no ML training required'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination 'metadata.json') -Encoding utf8
