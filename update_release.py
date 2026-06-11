"""Create GitHub release and upload katago.exe"""
import json, urllib.request, os, sys

# Read token
token = None
for path in [
    r'C:\Users\bzhao\AppData\Local\hermes\.env',
    os.path.expanduser('~/.hermes/.env'),
]:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('GITHUB_TOKEN') and '=' in line and '***' not in line:
                    _, val = line.split('=', 1)
                    token = val.strip()
                    break
    except:
        pass
    if token:
        break

if not token:
    print("ERROR: No GITHUB_TOKEN")
    sys.exit(1)

# Create release
data = json.dumps({
    'tag_name': 'v1.16.5-kab2-daemon',
    'name': 'KataGo v1.16.5-kab2 with daemon mode and game-id stream protocol',
    'body': (
        '## Changes\n\n'
        '- **Daemon mode** (`-daemon`): persistent process, jobs fed via stdin\n'
        '  (one line = games.csv path; `reset` clears NN caches; `quit` exits).\n'
        '  Models load once; per-job latency drops to analysis time itself.\n'
        '- **Game-id stream protocol**: frames are now\n'
        '  `[1B side][4B idLen][game id][4B size][KAB2 payload]`; `0x01` marks\n'
        '  end of a daemon job, `0x00` end of stream. B/W frames pair reliably.\n'
        '- **Combined file output**: one compressed `<stem>.npz` per game\n'
        '  (`[4B B_size][B KAB2][4B W_size][W KAB2]`) plus `_meta.csv`.\n'
        '- **Fixes**: stream mode no longer writes `_meta.csv` to the working\n'
        '  directory; HumanSL no longer emits a bogus rank when evaluation\n'
        '  fails on the first move; non-Windows builds (mkdir portability).\n'
        '- **Lite mode** (`-no-trunk`): scalars-only (10 floats/move).\n'
        '- **HumanSL** (`-human-model`): rank annotation (20k\u20139d) \u2014 required\n'
        '  for KataRank training data.\n\n'
        'See `batch_analysis -help` for full usage; KAB2 format details in the\n'
        'KataRank README.'
    ),
    'draft': False,
    'prerelease': False,
}).encode()

req = urllib.request.Request(
    'https://api.github.com/repos/ahillzhao-msn/KataGo/releases',
    data=data,
    headers={
        'Authorization': f'token {token}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json'
    }
)
resp = json.loads(urllib.request.urlopen(req).read())
print(f"Release: {resp['html_url']}")
print(f"Upload URL: {resp['upload_url']}")

# Upload katago.exe
upload_url = resp['upload_url'].replace('{?name,label}', '?name=katago.exe')
exe_path = r'C:\Users\bzhao\katago-fork\cpp\katago.exe'
with open(exe_path, 'rb') as f:
    binary = f.read()

req = urllib.request.Request(
    upload_url,
    data=binary,
    headers={
        'Authorization': f'token {token}',
        'Content-Type': 'application/octet-stream',
        'Accept': 'application/vnd.github.v3+json'
    }
)
upload_resp = json.loads(urllib.request.urlopen(req).read())
print(f"Uploaded: {upload_resp.get('browser_download_url', 'OK')}")
