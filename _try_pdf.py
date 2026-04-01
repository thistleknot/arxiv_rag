import urllib.request

urls = [
    "https://arxiv.org/pdf/2412.17866",
    "https://arxiv.org/pdf/2412.17866v1",
    "https://arxiv.org/pdf/2412.17866.pdf",
    "https://arxiv.org/pdf/2412.17866v1.pdf",
    "https://export.arxiv.org/pdf/2412.17866",
]
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        r = urllib.request.urlopen(req, timeout=15)
        ct = r.headers.get("content-type", "?")
        cl = r.headers.get("content-length", "?")
        print(f"OK  {r.status}  type={ct}  len={cl}  {u}")
    except Exception as e:
        print(f"FAIL  {e}  {u}")
