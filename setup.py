import os
import shutil
import glob
import sysconfig
import pathlib
from setuptools import setup
from setuptools.command.build_py import build_py


CANN_SRC_RELATIVE_DIR = "python/site-packages/msmodelslim/pytorch/"
TARGET_PKG_NAME = "npuslim"
TARGET_OPS_SUBDIR = "cann_ops"
TARGET_OPS_PATH = os.path.join("src", TARGET_PKG_NAME, TARGET_OPS_SUBDIR)


class NpuSlimBuildPy(build_py):
    def run(self):
        self.execute(self._copy_cann_operators, (), msg="Copying CANN operators...")
        super().run()

    def _create_init_files(self, start_dir: pathlib.Path, root_dir: pathlib.Path):
        current_dir = start_dir
        while current_dir.is_relative_to(root_dir) and current_dir != root_dir.parent:
            init_file_path = current_dir / "__init__.py"
            if not init_file_path.exists():
                with open(init_file_path, "w") as f:
                    f.write(
                        f"# CANN Operators initialization file for {current_dir.name}\n"
                    )
            if current_dir == root_dir:
                break
            current_dir = current_dir.parent

    def _copy_cann_operators(self):
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not ext_suffix:
            raise EnvironmentError(
                "Could not determine Python extension suffix (ABI tag)."
            )

        abi_tag = ext_suffix.lstrip(".")
        print(f"Current Python ABI Tag is: {abi_tag}")

        ascend_home_path = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home_path or not os.path.isdir(ascend_home_path):
            raise EnvironmentError(
                "ASCEND_HOME_PATH environment variable not set or path does not exist. "
                "Please source the CANN environment before building 'npuslim'."
            )

        src_dir = os.path.join(ascend_home_path, CANN_SRC_RELATIVE_DIR)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        dst_root_dir = os.path.join(current_script_dir, TARGET_OPS_PATH)

        print(f"\n--- CANN Operator Collection ---")
        print(f"Source Root: {src_dir}")
        print(f"Target Root: {dst_root_dir}")

        os.makedirs(dst_root_dir, exist_ok=True)
        self._create_init_files(pathlib.Path(dst_root_dir), pathlib.Path(dst_root_dir))

        copied_files = 0
        search_pattern = os.path.join(src_dir, f"**/*.{abi_tag}")

        for file_path in glob.glob(search_pattern, recursive=True):
            relative_path = os.path.relpath(file_path, src_dir)
            target_file = os.path.join(dst_root_dir, relative_path)
            target_file_dir = os.path.dirname(target_file)

            os.makedirs(target_file_dir, exist_ok=True)
            self._create_init_files(
                pathlib.Path(target_file_dir), pathlib.Path(dst_root_dir)
            )

            if os.path.exists(target_file):
                try:
                    os.chmod(target_file, 0o644)
                except Exception as e:
                    file_name = os.path.basename(file_path)
                    print(
                        f"Warning: Could not change permissions on existing file {file_name}. Error: {e}. Skipping copy."
                    )
                    continue

            shutil.copy2(file_path, target_file)
            os.chmod(target_file, 0o550)
            copied_files += 1

        if copied_files == 0:
            print(
                f"Warning: No .so files matching ABI tag '{abi_tag}' found in {src_dir}. Check CANN and Python version compatibility."
            )
        else:
            print(
                f"Successfully copied {copied_files} .so operator files matching current Python version, preserving directory structure."
            )
        print("--------------------------------\n")


setup(
    cmdclass={
        "build_py": NpuSlimBuildPy,
    },
    package_data={
        TARGET_PKG_NAME: [
            f"{TARGET_OPS_SUBDIR}/**/*.so",
        ],
    },
)
