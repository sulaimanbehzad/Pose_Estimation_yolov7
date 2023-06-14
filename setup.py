import cx_Freeze
import sys

base = None

if sys.platform == 'win32':
    base = "Win32GUI"
sys.setrecursionlimit(6000)
executables = [cx_Freeze.Executable("Detect_Coordinates.py", base=base, target_name="Detect Coordinates",
                                    icon="data\\Icons\\reality_converter_macos_bigsur_icon_189795.ico")]

cx_Freeze.setup(
    name="Detect Coordinates",
    options={"build_exe": {"packages": ["tkinter", "torch", "llvmlite.binding", "seaborn"],
                           "include_files": ["yolov7-w6-pose.pt", "models/yolo.py", "models/experimental.py",
                                             "models/common.py"], "excludes": [""], 'includes': ['models']}},
    version="1.0",
    description="DESCRIBE YOUR PROGRAM",
    executables=executables
)
