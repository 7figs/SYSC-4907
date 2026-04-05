import http.server
import socketserver
import os

PORT = 8000
MOVIE_BASE_DIR = "."
FALLBACK_DIR_NAME = "movies/PLACEHOLDER"


class FileHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        original_path = super().translate_path(path)

        if os.path.exists(original_path) and not os.path.isdir(original_path):
            return original_path

        filename = os.path.basename(original_path)

        fallback_path = os.path.join(os.getcwd(), FALLBACK_DIR_NAME, filename)

        if os.path.exists(fallback_path):
            print(
                f"[RECOVERED] Missing {filename} -> Serving from {FALLBACK_DIR_NAME}")
            return fallback_path

        print(f"[NOT FOUND] Checked: {original_path}")
        print(f"            Checked Fallback: {fallback_path}")
        return original_path

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


mimetypes = http.server.SimpleHTTPRequestHandler.extensions_map
mimetypes.update({
    '.m3u8': 'application/vnd.apple.mpegurl',
    '.ts':   'video/mp2t',
})

print(f"Server started. Root: {os.getcwd()}")
print(
    f"Fallback folder expected: {os.path.join(os.getcwd(), FALLBACK_DIR_NAME)}")
print("------------------------------------------------")

with socketserver.TCPServer(("", PORT), FileHandler) as server:
    server.serve_forever()
