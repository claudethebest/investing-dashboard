"""
Tiny dependency-free web server for the grid backtest dashboard.

Routes:
  /                -> dashboard.html
  /api/results     -> results.json (precomputed backtest; run build_data.py to refresh)
  /api/price       -> LIVE BTC price from both exchanges (proxied server-side to
                      dodge browser CORS), refreshed by the page every 5s.

Run:  python serve.py     then open  http://localhost:8000
"""

import os
import json
import http.server
import socketserver
import requests   # uses certifi CA bundle; system urllib fails on the local SSL proxy

PORT = 8000

PRICE_FEEDS = {
    "BTCUSDT": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
    "BTCTHB":  "https://api.binance.th/api/v1/ticker/price?symbol=BTCTHB",
}


def live_prices():
    out = {}
    for sym, url in PRICE_FEEDS.items():
        try:
            out[sym] = float(requests.get(url, timeout=6).json()["price"])
        except Exception:
            pass
    return out


def stock_quotes(tickers=("SPY", "QQQ", "VT")):
    out = {}
    for t in tickers:
        try:
            u = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=1d&interval=1d"
            j = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).json()
            out[t] = float(j["chart"]["result"][0]["meta"]["regularMarketPrice"])
        except Exception:
            pass
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("index.html", "rb") as f:
                    self._send(f.read(), "text/html")
            except FileNotFoundError:
                self._send("index.html missing", "text/plain", 404)
        elif self.path.startswith("/stocks"):
            try:
                with open("stocks.html", "rb") as f:
                    self._send(f.read(), "text/html")
            except FileNotFoundError:
                self._send("stocks.html missing", "text/plain", 404)
        elif self.path.startswith("/api/stocks"):
            try:
                with open("stocks.json", "rb") as f:
                    self._send(f.read())
            except FileNotFoundError:
                self._send('{"error":"run stock_data.py first"}', code=503)
        elif self.path.startswith("/api/stockquote"):
            self._send(json.dumps(stock_quotes()))
        elif self.path.startswith("/api/thaifunds"):
            try:
                with open("thaifunds.json", "rb") as f:
                    self._send(f.read())
            except FileNotFoundError:
                self._send('{"error":"thaifunds.json missing"}', code=503)
        elif self.path.startswith("/api/results"):
            try:
                with open("results.json", "rb") as f:
                    self._send(f.read())
            except FileNotFoundError:
                self._send('{"error":"run build_data.py first"}', code=503)
        elif self.path.startswith("/api/price"):
            self._send(json.dumps(live_prices()))
        elif self.path.endswith(".png"):
            fn = os.path.basename(self.path.lstrip("/"))   # cwd only, no traversal
            try:
                with open(fn, "rb") as f:
                    self._send(f.read(), "image/png")
            except FileNotFoundError:
                self._send("not found", "text/plain", 404)
        elif self.path.endswith(".json"):                  # static JSON for the pages
            fn = os.path.basename(self.path.lstrip("/"))
            try:
                with open(fn, "rb") as f:
                    self._send(f.read())
            except FileNotFoundError:
                self._send("not found", "text/plain", 404)
        elif self.path.endswith(".html"):                  # e.g. /stocks.html, /index.html
            fn = os.path.basename(self.path.lstrip("/"))
            try:
                with open(fn, "rb") as f:
                    self._send(f.read(), "text/html")
            except FileNotFoundError:
                self._send("not found", "text/plain", 404)
        else:
            self._send("not found", "text/plain", 404)

    def log_message(self, *a):    # quiet console
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"dashboard live at  http://localhost:{PORT}")
        httpd.serve_forever()
