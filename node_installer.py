"""Auto-detect and install missing ComfyUI custom nodes from workflow JSON.

Uses ComfyUI-Manager's API to resolve class_type → git repo. The Manager
applies preemption rules and we additionally rank by star count to pick
the correct repo when multiple claim the same node.

Flow:
    1. Extract all class_type values from workflow
    2. Query ComfyUI /object_info for installed node types
    3. Diff → missing class_types
    4. Query ComfyUI-Manager /customnode/getmappings for node→repo map
    5. Query ComfyUI-Manager /customnode/getlist for repo star counts
    6. For each missing node, pick the repo with the most stars
    7. git clone + pip install deps
    8. Restart ComfyUI
"""

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
import urllib.error

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/ComfyUI")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")

# Fallback: raw GitHub map used when ComfyUI-Manager is not available
NODE_MAP_CACHE = "/runpod-volume/.node-map-cache.json"
NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
NODE_MAP_MAX_AGE = 3600  # refresh cache after 1h


def extract_class_types(workflow: dict) -> set[str]:
    """Extract all unique class_type values from an API-format workflow."""
    return {
        node["class_type"]
        for node in workflow.values()
        if isinstance(node, dict) and "class_type" in node
    }


def get_installed_node_types() -> set[str]:
    """Query ComfyUI /object_info for all registered node types."""
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=10) as r:
            data = json.loads(r.read())
        return set(data.keys())
    except Exception as e:
        print(f"[node_installer] WARNING: Could not query /object_info: {e}", flush=True)
        return set()


def _get_manager_mappings() -> dict | None:
    """Fetch node mappings from ComfyUI-Manager with preemptions applied.

    Returns the mappings dict or None if Manager is not available.
    """
    try:
        url = f"{COMFY_URL}/customnode/getmappings?mode=nickname"
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[node_installer] ComfyUI-Manager not available: {e}", flush=True)
        return None


def _get_manager_pack_stars() -> dict[str, int]:
    """Fetch star counts for all node packs from ComfyUI-Manager.

    Returns {pack_id_or_url: star_count}.
    """
    try:
        url = f"{COMFY_URL}/customnode/getlist?mode=local&skip_update=true"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"[node_installer] Could not fetch pack list: {e}", flush=True)
        return {}

    stars = {}
    for pid, info in data.get("node_packs", {}).items():
        s = info.get("stars", 0) or 0
        stars[pid] = s
        repo = info.get("repository", "")
        if repo:
            stars[repo] = s
    return stars


def _get_fallback_node_map() -> dict:
    """Fetch raw extension-node-map from GitHub (no preemptions).

    Used only when ComfyUI-Manager is not installed/running.
    """
    if os.path.exists(NODE_MAP_CACHE):
        age = time.time() - os.path.getmtime(NODE_MAP_CACHE)
        if age < NODE_MAP_MAX_AGE:
            with open(NODE_MAP_CACHE) as f:
                return json.load(f)

    print("[node_installer] Fetching extension-node-map from GitHub...", flush=True)
    try:
        with urllib.request.urlopen(NODE_MAP_URL, timeout=15) as r:
            data = json.loads(r.read())
        os.makedirs(os.path.dirname(NODE_MAP_CACHE), exist_ok=True)
        with open(NODE_MAP_CACHE, "w") as f:
            json.dump(data, f)
        print(f"[node_installer] Cached node map ({len(data)} entries)", flush=True)
        return data
    except Exception as e:
        print(f"[node_installer] WARNING: Could not fetch node map: {e}", flush=True)
        if os.path.exists(NODE_MAP_CACHE):
            with open(NODE_MAP_CACHE) as f:
                return json.load(f)
        return {}


def _build_fallback_node_to_repo() -> dict[str, str]:
    """Build class_type → git URL from the live GitHub extension-node-map.

    The Manager DB baked into the image goes stale between rebuilds, so a node
    registered after the last image build is absent from it. This live map is
    fetched fresh (1h cache) and covers those newer nodes.
    """
    raw_map = _get_fallback_node_map()
    node_to_repo: dict[str, str] = {}
    for repo_url, info in raw_map.items():
        if not isinstance(info, list) or len(info) == 0:
            continue
        nodes = info[0] if isinstance(info[0], list) else []
        for n in nodes:
            if n not in node_to_repo:
                node_to_repo[n] = repo_url
    return node_to_repo


def _build_node_to_repo(mappings: dict, pack_stars: dict) -> dict[str, str]:
    """Build a class_type → repo lookup, picking the most popular repo on conflict.

    Args:
        mappings: {repo_id_or_url: [[class_types...], {metadata}]}
        pack_stars: {repo_id_or_url: star_count}

    Returns:
        {class_type: repo_url_or_id}
    """
    # Collect all repos that claim each node
    node_candidates = {}
    for repo, info in mappings.items():
        if not isinstance(info, list) or len(info) == 0:
            continue
        nodes = info[0] if isinstance(info[0], list) else []
        for n in nodes:
            node_candidates.setdefault(n, []).append(repo)

    # For each node, pick the repo with the most stars
    node_to_repo = {}
    for node, repos in node_candidates.items():
        if len(repos) == 1:
            node_to_repo[node] = repos[0]
        else:
            best = max(repos, key=lambda r: pack_stars.get(r, 0))
            node_to_repo[node] = best

    return node_to_repo


def resolve_repos(missing_types: set[str], node_to_repo: dict) -> dict[str, list[str]]:
    """Map missing class_types to git repo URLs/IDs.

    Returns {repo: [class_type, ...]} for repos that need installing.
    """
    repos = {}
    for ct in missing_types:
        repo = node_to_repo.get(ct)
        if repo:
            repos.setdefault(repo, []).append(ct)
    return repos


def _resolve_repo_url(repo_id: str, mappings: dict) -> str:
    """Resolve a ComfyRegistry ID to a git URL for cloning.

    Manager mappings use CNR IDs (e.g. 'comfyui-videohelpersuite') as keys
    instead of full URLs. We need to find the git URL for cloning.
    """
    if repo_id.startswith("http"):
        return repo_id

    # Check if the mapping metadata has install info
    info = mappings.get(repo_id, [])
    if len(info) > 1 and isinstance(info[1], dict):
        # Some entries have 'title_aux' with the repo name
        title = info[1].get("title_aux", "")
        if title:
            # Try common GitHub URL pattern
            url = f"https://github.com/search?q={repo_id}"

    # Try fetching pack info from Manager for the git URL
    try:
        url = f"{COMFY_URL}/customnode/getlist?mode=local&skip_update=true"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        pack = data.get("node_packs", {}).get(repo_id, {})
        git_url = pack.get("repository", "")
        if git_url:
            return git_url
    except Exception:
        pass

    # Last resort: assume GitHub with the ID as repo name
    print(f"[node_installer] WARNING: Could not resolve git URL for '{repo_id}', skipping", flush=True)
    return ""


def install_repo(repo_url: str, force_deps: bool = False) -> bool:
    """Clone a custom node repo and install its dependencies.

    Args:
        repo_url: Git URL of the custom node repo.
        force_deps: If True, reinstall deps even if repo dir already exists.
                    Used when the repo exists but its nodes failed to import.
    """
    if not repo_url:
        return False

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target = os.path.join(CUSTOM_NODES_DIR, repo_name)

    if os.path.exists(target) and not force_deps:
        print(f"[node_installer] {repo_name} already exists, skipping clone", flush=True)
        return False

    if os.path.exists(target) and force_deps:
        print(f"[node_installer] {repo_name} exists but nodes not loaded — reinstalling deps", flush=True)
    else:
        print(f"[node_installer] Installing {repo_name} from {repo_url}", flush=True)
        # Clone
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"[node_installer] ERROR cloning {repo_name}: {result.stderr[:500]}", flush=True)
            return False

    # Install requirements.txt if present
    req_file = os.path.join(target, "requirements.txt")
    if os.path.exists(req_file):
        print(f"[node_installer] Installing deps for {repo_name}...", flush=True)
        subprocess.run(
            ["pip", "install", "-q", "-r", req_file],
            capture_output=True, text=True, timeout=300,
        )

    # Run install.py if present
    install_script = os.path.join(target, "install.py")
    if os.path.exists(install_script):
        print(f"[node_installer] Running install.py for {repo_name}...", flush=True)
        subprocess.run(
            ["python3", install_script],
            capture_output=True, text=True, timeout=120,
            cwd=target,
        )

    print(f"[node_installer] {repo_name} installed successfully", flush=True)
    return True


def restart_comfyui() -> bool:
    """Kill ComfyUI and restart it fresh. Wait until ready."""
    print("[node_installer] Restarting ComfyUI...", flush=True)

    # Find and kill ComfyUI process
    result = subprocess.run(
        ["pkill", "-f", "python3 main.py"],
        capture_output=True, text=True,
    )
    time.sleep(2)

    # Start ComfyUI in background
    comfy_port = os.environ.get("COMFYUI_PORT", "8188")
    cmd = [
        "python3", "main.py",
        "--listen", "0.0.0.0",
        "--port", comfy_port,
        "--disable-auto-launch",
        "--disable-metadata",
    ]
    extra_paths = os.path.join(COMFYUI_DIR, "extra_model_paths.yaml")
    if os.path.exists(extra_paths):
        cmd += ["--extra-model-paths-config", extra_paths]
    subprocess.Popen(
        cmd,
        cwd=COMFYUI_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for ready
    max_wait = 120
    waited = 0
    while waited < max_wait:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as r:
                r.read()
            print(f"[node_installer] ComfyUI ready after {waited}s", flush=True)
            return True
        except Exception:
            time.sleep(2)
            waited += 2

    print(f"[node_installer] ERROR: ComfyUI failed to restart within {max_wait}s", flush=True)
    return False


def parse_missing_node_from_error(error_msg: str) -> str | None:
    """Extract missing node class_type from ComfyUI error message."""
    # Pattern: "Cannot execute because node X does not exist."
    m = re.search(r"node (\S+) does not exist", error_msg)
    return m.group(1) if m else None


def ensure_nodes(workflow: dict, max_retries: int = 3, progress_fn=None) -> list[str]:
    """Check workflow for missing nodes and install them.

    Resolution strategy:
    1. Try ComfyUI-Manager API (preemptions + star-count ranking)
    2. Fall back to raw GitHub extension-node-map if Manager unavailable

    Args:
        workflow: ComfyUI API-format workflow dict.
        max_retries: Max retry attempts for node installation.
        progress_fn: Optional callback(message: str) for progress updates.

    Returns list of newly installed repo names.
    """
    def _progress(msg: str) -> None:
        if progress_fn:
            progress_fn(msg)

    installed_types = get_installed_node_types()
    required_types = extract_class_types(workflow)
    missing = required_types - installed_types

    if not missing:
        return []

    print(f"[node_installer] Missing node types: {missing}", flush=True)
    _progress(f"Resolving {len(missing)} missing custom node(s)")

    # Try ComfyUI-Manager API first (preemptions applied, star ranking)
    mappings = _get_manager_mappings()
    if mappings:
        pack_stars = _get_manager_pack_stars()
        node_to_repo = _build_node_to_repo(mappings, pack_stars)
        repos = resolve_repos(missing, node_to_repo)

        # Resolve CNR IDs to git URLs
        resolved_repos = {}
        for repo_id, types in repos.items():
            git_url = _resolve_repo_url(repo_id, mappings)
            if git_url:
                resolved_repos[git_url] = types
        repos = resolved_repos

        # Supplement with the live GitHub map for anything the (image-baked,
        # possibly stale) Manager DB couldn't resolve — e.g. a node registered
        # after the last image build.
        resolved_now = {t for types in repos.values() for t in types}
        still_missing = missing - resolved_now
        if still_missing:
            print(f"[node_installer] {len(still_missing)} type(s) unresolved by Manager, "
                  f"trying live extension-node-map", flush=True)
            fallback_map = _build_fallback_node_to_repo()
            node_to_repo = {**fallback_map, **node_to_repo}  # Manager wins on overlap
            for repo_url, types in resolve_repos(still_missing, fallback_map).items():
                repos.setdefault(repo_url, []).extend(types)
    else:
        # Fallback: raw extension-node-map (no preemptions, no star ranking)
        print("[node_installer] Falling back to raw extension-node-map", flush=True)
        node_to_repo = _build_fallback_node_to_repo()
        repos = resolve_repos(missing, node_to_repo)

    if not repos:
        resolved_types = set(node_to_repo.keys())
        unresolved = missing - resolved_types
        if unresolved:
            print(f"[node_installer] WARNING: Could not resolve repos for: {unresolved}", flush=True)
        return []

    # Report unresolved types
    resolved_in_repos = {t for types in repos.values() for t in types}
    unresolved = missing - resolved_in_repos
    if unresolved:
        print(f"[node_installer] WARNING: Could not resolve repos for: {unresolved}", flush=True)

    # Install all missing repos (force deps if dir exists but nodes aren't loaded)
    installed_repos = []
    total = len(repos)
    for i, (repo_url, types) in enumerate(repos.items()):
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        target = os.path.join(CUSTOM_NODES_DIR, repo_name)
        dir_exists = os.path.exists(target)
        print(f"[node_installer] {repo_name} provides: {types}", flush=True)
        _progress(f"Installing {repo_name} ({i+1}/{total})")
        if install_repo(repo_url, force_deps=dir_exists):
            installed_repos.append(repo_name)

    # Restart ComfyUI to pick up new nodes
    if installed_repos:
        _progress(f"Restarting ComfyUI after installing {len(installed_repos)} node(s)")
        if not restart_comfyui():
            raise RuntimeError("Failed to restart ComfyUI after installing nodes")

    return installed_repos
