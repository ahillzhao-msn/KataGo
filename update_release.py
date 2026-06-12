"""
Upload a local katago binary to an existing (or new) GitHub release.

Usage:
    # Upload to an existing release tag
    python update_release.py --tag v1.16.5-kab2 --binary cpp/katago.exe

    # Create release then upload (if tag doesn't exist yet)
    python update_release.py --tag v1.16.5-kab2 --binary cpp/katago.exe --create

    # Custom asset name
    python update_release.py --tag v1.16.5-kab2 --binary cpp/katago.exe --name katago-cuda-windows.exe

Environment:
    GITHUB_TOKEN  Personal access token with repo write scope.
                  Set in your shell or in ~/.hermes/.env / %APPDATA%/hermes/.env
"""
import argparse
import json
import os
import sys
import urllib.request


REPO = "ahillzhao-msn/KataGo"
API = "https://api.github.com"


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    for candidate in [
        os.path.join(os.environ.get("APPDATA", ""), "hermes", ".env"),
        os.path.expanduser("~/.hermes/.env"),
        os.path.expanduser("~/.env"),
    ]:
        try:
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GITHUB_TOKEN") and "=" in line and "***" not in line:
                        _, val = line.split("=", 1)
                        tok = val.strip().strip('"').strip("'")
                        if tok:
                            return tok
        except OSError:
            pass
    print("ERROR: GITHUB_TOKEN not set and not found in env files.", file=sys.stderr)
    sys.exit(1)


def _req(method: str, url: str, token: str, data=None, content_type="application/json"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code} {method} {url}: {body}", file=sys.stderr)
        sys.exit(1)


def get_release(token: str, tag: str) -> dict | None:
    url = f"{API}/repos/{REPO}/releases/tags/{tag}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def create_release(token: str, tag: str, name: str | None) -> dict:
    data = json.dumps({
        "tag_name": tag,
        "name": name or f"KataGo {tag} (KAB2 fork)",
        "draft": False,
        "prerelease": "-rc" in tag or "-beta" in tag,
        "generate_release_notes": True,
    }).encode()
    print(f"Creating release {tag}...")
    return _req("POST", f"{API}/repos/{REPO}/releases", token, data)


def upload_asset(token: str, upload_url: str, asset_name: str, binary_path: str):
    url = upload_url.replace("{?name,label}", f"?name={asset_name}")
    with open(binary_path, "rb") as f:
        binary = f.read()
    size = len(binary)
    print(f"Uploading {asset_name} ({size / 1024 / 1024:.1f} MB)...")
    result = _req("POST", url, token, data=binary, content_type="application/octet-stream")
    print(f"  → {result.get('browser_download_url', 'uploaded')}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v1.16.5-kab2")
    parser.add_argument("--binary", required=True, help="Path to the katago binary to upload")
    parser.add_argument("--name", help="Asset filename on the release (default: basename of --binary)")
    parser.add_argument("--release-name", help="Release title (only used with --create)")
    parser.add_argument("--create", action="store_true", help="Create the release if it does not exist")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    asset_name = args.name or os.path.basename(args.binary)
    token = _token()

    release = get_release(token, args.tag)
    if release is None:
        if not args.create:
            print(f"ERROR: Release {args.tag} not found. Use --create to create it.", file=sys.stderr)
            sys.exit(1)
        release = create_release(token, args.tag, args.release_name)
    else:
        print(f"Found release: {release['html_url']}")

    upload_asset(token, release["upload_url"], asset_name, args.binary)
    print("Done.")


if __name__ == "__main__":
    main()
