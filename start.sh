#!/bin/bash
set -euo pipefail

COMFYUI_DIR="/ComfyUI"
COMFYUI_PORT=8188
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFY_LOG="/tmp/comfyui_startup.log"

# Stamp the runtime commit at every boot so we can tell from the worker log
# which SHA is actually running. Critical when diagnosing FlashBoot snapshots
# vs true cold starts (per bead 6i0).
RUNTIME_SHA="$(git -C "$RUNTIME_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUNTIME_SUBJ="$(git -C "$RUNTIME_DIR" log -1 --pretty=%s 2>/dev/null || echo '?')"
echo "[start] runtime commit ${RUNTIME_SHA}: ${RUNTIME_SUBJ}"
export RUNTIME_COMMIT="${RUNTIME_SHA}: ${RUNTIME_SUBJ}"

echo "[start] Booting ComfyUI from $COMFYUI_DIR (baked in image)..."

# Verify ComfyUI exists
if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo "[start] ERROR: ComfyUI not found at $COMFYUI_DIR"
    exit 1
fi

# --- Force ComfyUI-Manager offline mode (skip remote fetches, security scan, alembic) ---
MANAGER_CONFIG_DIR="$COMFYUI_DIR/user/__manager"
mkdir -p "$MANAGER_CONFIG_DIR"
if [ ! -f "$MANAGER_CONFIG_DIR/config.ini" ] || ! grep -q "network_mode = offline" "$MANAGER_CONFIG_DIR/config.ini" 2>/dev/null; then
    cat > "$MANAGER_CONFIG_DIR/config.ini" <<'MGREOF'
[default]
network_mode = offline
security_level = normal
MGREOF
    echo "[start] Set ComfyUI-Manager to offline mode"
fi

# Build extra model paths flag — models live on the network volume, not in the image
EXTRA_PATHS_FLAG=""
if [ -f "$COMFYUI_DIR/extra_model_paths.yaml" ]; then
    # Patch in any missing model types without rebuilding the Docker image
    if ! grep -q "detection:" "$COMFYUI_DIR/extra_model_paths.yaml"; then
        sed -i '/^    vae:/i\    detection: detection' "$COMFYUI_DIR/extra_model_paths.yaml"
        echo "[start] Patched extra_model_paths.yaml with detection"
    fi
    if ! grep -q "frame_interpolation:" "$COMFYUI_DIR/extra_model_paths.yaml"; then
        sed -i '/^    vae:/i\    frame_interpolation: frame_interpolation' "$COMFYUI_DIR/extra_model_paths.yaml"
        echo "[start] Patched extra_model_paths.yaml with frame_interpolation"
    fi
    EXTRA_PATHS_FLAG="--extra-model-paths-config $COMFYUI_DIR/extra_model_paths.yaml"
    echo "[start] Using extra_model_paths.yaml for network volume models"
fi

# --- Hotpatch kijai/ComfyUI-WanAnimatePreprocess for ONNX dropdown bug (issue #32) ---
# folder_paths caches the detection folder's filename list under the default
# extensions (no .onnx) when ANY earlier-loading node calls
# get_filename_list("detection") before this node registers .onnx. Patch the
# node to add .onnx and invalidate the cache, idempotently.
WANIMATE_NODES="$COMFYUI_DIR/custom_nodes/ComfyUI-WanAnimatePreprocess/nodes.py"
if [ -f "$WANIMATE_NODES" ] && ! grep -q "filename_list_cache" "$WANIMATE_NODES"; then
    python3 - "$WANIMATE_NODES" <<'PYEOF' && echo "[start] Hotpatched WanAnimatePreprocess for ONNX dropdown bug"
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
trigger = 'folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))'
patch = '''
# --- onnx-dropdown hotpatch (kijai issue #32): register .onnx, invalidate cache ---
if "detection" in folder_paths.folder_names_and_paths:
    _paths, _exts = folder_paths.folder_names_and_paths["detection"]
    folder_paths.folder_names_and_paths["detection"] = (_paths, set(_exts) | {".onnx"})
    if hasattr(folder_paths, "filename_list_cache") and "detection" in folder_paths.filename_list_cache:
        del folder_paths.filename_list_cache["detection"]
    if hasattr(folder_paths, "cache_helper"):
        folder_paths.cache_helper.clear()
'''
if trigger in src and "filename_list_cache" not in src:
    p.write_text(src.replace(trigger, trigger + patch))
PYEOF
fi

# --- Start ComfyUI, tee output to log file for IMPORT FAILED detection ---
cd "$COMFYUI_DIR"
# Experimental performance flags (enable via EXPERIMENTAL=true env var)
PERF_FLAGS=""
if [ "${EXPERIMENTAL:-}" = "true" ]; then
    echo "[start] Experimental mode: enabling cublas_ops, flash-attention"
    # NOTE: fp8_matrix_mult causes corrupted output with Qwen Image models (ComfyUI #9190)
    # NOTE: --gpu-only removed — causes OOM on large workflows
    PERF_FLAGS="--fast cublas_ops --use-flash-attention"
fi

# SageAttention: enable the global flag ONLY if a real kernel actually launches on
# this GPU. The baked wheel is multi-arch (sm_80/sm_89/sm_120), but a scheduling
# mismatch or a bad build must NOT take ComfyUI down — fall back to default
# attention instead. Import success is NOT enough (that's what build-time checks
# miss); we launch an actual kernel and require it to complete.
SAGE_FLAG=""
if python3 - >/dev/null 2>&1 <<'SAGEPROBE'
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")
sageattn(q, q.clone(), q.clone())
torch.cuda.synchronize()
SAGEPROBE
then
    SAGE_FLAG="--use-sage-attention"
    echo "[start] SageAttention kernel probe passed — enabling --use-sage-attention"
else
    echo "[start] SageAttention kernel probe failed — using default attention"
fi

python3 main.py \
    --listen 0.0.0.0 \
    --port $COMFYUI_PORT \
    --disable-auto-launch \
    --disable-metadata \
    $PERF_FLAGS \
    $SAGE_FLAG \
    $EXTRA_PATHS_FLAG \
    > >(tee "$COMFY_LOG") 2>&1 &

COMFYUI_PID=$!
echo "[start] ComfyUI starting (PID: $COMFYUI_PID)..."

# Wait for ComfyUI to be ready
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
        echo "[start] ComfyUI ready after ${WAITED}s"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[start] ERROR: ComfyUI failed to start within ${MAX_WAIT}s"
    kill $COMFYUI_PID 2>/dev/null || true
    exit 1
fi

# --- Fix broken custom nodes (IMPORT FAILED) ---
# Parse ComfyUI startup log for nodes that failed to import,
# reinstall their deps, and restart ComfyUI if any were fixed.
BROKEN_NODES=$(grep -o 'IMPORT FAILED.*custom_nodes/[^"]*' "$COMFY_LOG" 2>/dev/null \
    | sed 's|.*/custom_nodes/||' | sort -u || true)

if [ -n "$BROKEN_NODES" ]; then
    echo "[start] Found broken custom nodes:"
    NEEDS_RESTART=false
    CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"

    while IFS= read -r node_name; do
        req_file="$CUSTOM_NODES_DIR/$node_name/requirements.txt"
        if [ -f "$req_file" ]; then
            echo "[start]   -> $node_name: reinstalling deps..."
            pip install -q -r "$req_file" 2>/dev/null && NEEDS_RESTART=true \
                || echo "[start]   WARNING: deps install failed for $node_name"
        else
            echo "[start]   -> $node_name: no requirements.txt, skipping"
        fi
    done <<< "$BROKEN_NODES"

    if $NEEDS_RESTART; then
        echo "[start] Restarting ComfyUI to reload fixed nodes..."
        kill $COMFYUI_PID 2>/dev/null || true
        sleep 2

        cd "$COMFYUI_DIR"
        python3 main.py \
            --listen 0.0.0.0 \
            --port $COMFYUI_PORT \
            --disable-auto-launch \
            --disable-metadata \
            $PERF_FLAGS \
            $SAGE_FLAG \
            $EXTRA_PATHS_FLAG \
            &
        COMFYUI_PID=$!

        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
                echo "[start] ComfyUI restarted after ${WAITED}s"
                break
            fi
            sleep 2
            WAITED=$((WAITED + 2))
        done

        # Invalidate deps stamp so start_script.sh reinstalls next time
        rm -f /runpod-volume/.custom-node-deps-stamp
    fi
else
    echo "[start] All custom nodes loaded OK"
fi

# Start the RunPod worker handler
echo "[start] Starting RunPod handler..."
exec python3 "$RUNTIME_DIR/worker.py"
