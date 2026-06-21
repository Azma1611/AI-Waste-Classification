# Simple PowerShell HTTP Server using .NET constructor
$port = 8080
try {
    $listener = [System.Net.HttpListener]::new()
    $listener.Prefixes.Add("http://127.0.0.1:$port/")
    $listener.Start()
    Write-Host "EcoScan Server started. Listening on http://127.0.0.1:$port/ ..."
} catch {
    Write-Host "Failed to start listener: $_"
    exit 1
}

$root = "c:\Main\new one"

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        
        $url = $request.Url.LocalPath
        if ($url -eq "/") { $url = "/index.html" }
        
        $filePath = Join-Path $root $url
        
        # Check if file exists
        if (Test-Path $filePath -PathType Leaf) {
            $bytes = [System.IO.File]::ReadAllBytes($filePath)
            
            $ext = [System.IO.Path]::GetExtension($filePath)
            $contentType = "application/octet-stream"
            switch ($ext) {
                ".html" { $contentType = "text/html; charset=utf-8" }
                ".css"  { $contentType = "text/css; charset=utf-8" }
                ".js"   { $contentType = "application/javascript; charset=utf-8" }
                ".json" { $contentType = "application/json; charset=utf-8" }
                ".png"  { $contentType = "image/png" }
                ".jpg"  { $contentType = "image/jpeg" }
                ".jpeg" { $contentType = "image/jpeg" }
                ".gif"  { $contentType = "image/gif" }
                ".webp" { $contentType = "image/webp" }
                ".svg"  { $contentType = "image/svg+xml; charset=utf-8" }
                ".ico"  { $contentType = "image/x-icon" }
            }
            
            $response.ContentType = $contentType
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $response.StatusCode = 404
            $response.ContentType = "text/plain; charset=utf-8"
            $errorMessage = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found")
            $response.OutputStream.Write($errorMessage, 0, $errorMessage.Length)
        }
        $response.Close()
    }
} catch {
    Write-Host "Error occurred: $_"
} finally {
    if ($null -ne $listener) {
        $listener.Stop()
    }
    Write-Host "EcoScan Server stopped."
}
