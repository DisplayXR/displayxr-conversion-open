# PyInstaller spec: the SHARP export tool (scripts/export_sharp.py frozen, onedir).
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

REPO = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas = [(os.path.join(REPO, "packs", "dxr-open-sharp.models.json"), "packs")]
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
    [os.path.join(REPO, "scripts", "export_sharp.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + ["scipy", "scipy.spatial.transform", "PIL.Image", "torchvision"],
    excludes=["matplotlib", "tkinter", "IPython", "jupyter", "pytest"],
    module_collection_mode={"torch": "pyz+py", "torchvision": "pyz+py", "onnxscript": "pyz+py",
                            "timm": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="displayxr-export-sharp",
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="displayxr-export-sharp", upx=False)
