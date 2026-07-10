#!/usr/bin/env python
"""通过 curl + HF API 直接下载模型到 HF cache (绕过 hf CLI 与 hf-mirror 的 metadata 不兼容)。
用法: python curl_hf_download.py <repo_id> [target_dir]
  - 不给 target_dir: 下载到 HF hub cache 的标准 snapshot 结构 (供 from_pretrained 直接用 repo_id 加载)
"""
import os, sys, json, subprocess, hashlib

MIRROR = "https://hf-mirror.com"
repo = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else None

# 1) 取文件列表 + commit sha (优先官方 API, 失败再用镜像)
def get_meta(repo):
    for base in ["https://huggingface.co", MIRROR]:
        try:
            out = subprocess.check_output(
                ["curl", "-sL", f"{base}/api/models/{repo}"], timeout=30).decode()
            d = json.loads(out)
            files = [s["rfilename"] for s in d["siblings"]]
            sha = d.get("sha", "main")
            return files, sha
        except Exception as e:
            print(f"  meta via {base} failed: {e}")
    raise SystemExit("cannot get repo meta")

files, sha = get_meta(repo)
print(f"repo={repo} sha={sha} files={len(files)}")

if target is None:
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    rname = "models--" + repo.replace("/", "--")
    snap = os.path.join(cache, rname, "snapshots", sha)
    blobs = os.path.join(cache, rname, "blobs")
    refs = os.path.join(cache, rname, "refs")
    os.makedirs(snap, exist_ok=True); os.makedirs(blobs, exist_ok=True); os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as f:
        f.write(sha)
    dest_dir = snap
    use_symlink = True
else:
    os.makedirs(target, exist_ok=True)
    dest_dir = target
    use_symlink = False

for f in files:
    out = os.path.join(dest_dir, f)
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(f) else None
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  {f} exists"); continue
    url = f"{MIRROR}/{repo}/resolve/{sha}/{f}"
    tmp = out + ".part"
    code = subprocess.call(["curl", "-sL", "--fail", "-o", tmp, url])
    if code != 0 or not os.path.exists(tmp):
        print(f"  {f} -> FAIL(code={code})");
        if os.path.exists(tmp): os.remove(tmp)
        continue
    os.replace(tmp, out)
    print(f"  {f} -> OK ({os.path.getsize(out)} B)")

print("DONE", repo)
