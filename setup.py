import os
import subprocess
import sys
import warnings
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


TARGET_PKG_NAME = "npuslim"
OPS_DIR = os.path.join("src", TARGET_PKG_NAME, "ops")


def _build_custom_ops():
    """Discover and build all ops/*/build.py via subprocess."""
    if os.environ.get("NPUSLIM_SKIP_OPS"):
        print("\n--- Custom Operator Build ---")
        print("  Skipped (NPUSLIM_SKIP_OPS is set)")
        print("------------------------------\n")
        return

    if not os.path.isdir(OPS_DIR):
        return

    print("\n--- Custom Operator Build ---")
    for entry in sorted(os.listdir(OPS_DIR)):
        op_dir = os.path.abspath(os.path.join(OPS_DIR, entry))
        build_file = os.path.join(op_dir, "build.py")
        if not os.path.isfile(build_file):
            continue

        print(f"  [{entry}] building ...")
        try:
            result = subprocess.run(
                [sys.executable, build_file],
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode == 0:
                print(f"  [{entry}] OK")
            else:
                print(f"  [{entry}] FAILED (exit {result.returncode})")
                if result.stderr:
                    print(result.stderr)
        except Exception as e:
            warnings.warn(
                f"[{entry}] build failed: {e}. "
                "Other npuslim features remain usable.",
                stacklevel=2,
            )
    print("------------------------------\n")


class NpuSlimBuildPy(build_py):
    def run(self):
        self.execute(_build_custom_ops, (), msg="Building custom operators...")
        super().run()


class NpuSlimDevelop(develop):
    def run(self):
        self.execute(_build_custom_ops, (), msg="Building custom operators...")
        super().run()


setup(
    cmdclass={
        "build_py": NpuSlimBuildPy,
        "develop": NpuSlimDevelop,
    },
    package_data={
        TARGET_PKG_NAME: [
            "ops/**/*.so",
        ],
    },
)
