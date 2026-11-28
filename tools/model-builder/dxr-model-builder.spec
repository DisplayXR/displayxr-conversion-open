# PyInstaller spec: the DisplayXR open model builder (tools/model-builder/dxr_model_builder.py,
# onedir). It carries every pinned dependency of its recipes, so its output is byte-identical to
# the pack's hashes, and apple/ml-sharp's src/ (re-checked against the pinned tree hash at run time).
# Contract: displayxr-runtime docs/specs/runtime/conversion-provider-discovery.md section 5.6.2.
import os
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

REPO = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import export_sharp  # noqa: E402

# apple/ml-sharp at the pinned commit, from its source archive, refused unless the src/ tree hash matches.
ml_work = os.path.join(REPO, "build", "model-builder")
os.makedirs(ml_work, exist_ok=True)
ml_src = export_sharp.prepare_ml_sharp_archive(ml_work)
ml_root = os.path.dirname(ml_src)

datas = [(os.path.join(REPO, "packs", "dxr-open-sharp.models.json"), "packs")]
for dirpath, _, names in os.walk(ml_src):
    if "__pycache__" in dirpath:
        continue
    rel = os.path.relpath(dirpath, ml_root)
    datas += [(os.path.join(dirpath, n), os.path.join("ml-sharp", rel)) for n in names]
datas += [(os.path.join(ml_root, n), "ml-sharp") for n in ("LICENSE", "ACKNOWLEDGEMENTS")
          if os.path.isfile(os.path.join(ml_root, n))]

binaries, hidden = [], []
for pkg in ("torchvision", "timm", "onnx", "onnxruntime", "onnxscript", "onnx_ir", "onnxconverter_common",
            "ml_dtypes", "plyfile", "safetensors", "huggingface_hub"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
for dist in ("torch", "torchvision", "onnx", "onnxruntime", "onnxconverter-common", "onnxscript",
             "onnx-ir", "numpy", "timm", "protobuf", "ml-dtypes", "safetensors", "huggingface-hub",
             "pillow", "scipy", "plyfile"):
    datas += copy_metadata(dist)

a = Analysis(
    [os.path.join(SPECPATH, "dxr_model_builder.py")],
    pathex=[os.path.join(REPO, "scripts")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + ["export_sharp", "scipy", "scipy.spatial.transform", "PIL.Image", "torchvision"],
    excludes=["matplotlib", "tkinter", "IPython", "jupyter", "pytest"],
    module_collection_mode={"torch": "pyz+py", "torchvision": "pyz+py", "onnxscript": "pyz+py",
                            "timm": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="dxr-model-builder",
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="dxr-model-builder", upx=False)
