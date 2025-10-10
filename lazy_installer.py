#!/usr/bin/env python3
"""
Lazy Dependency Installer for Anime Dubbing Service

This module handles runtime detection and installation of heavy dependencies
(torch, audio-separator) to avoid including them in build binaries.

Features:
- Environment detection (CPU/CUDA/ROCm)
- Dynamic package installation using uv (preferred) or pip
- Caching in deps/ directory
- Minimal changes to existing code
"""

import importlib
import os
import subprocess
import sys
import platform
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


class LazyInstaller:
    """Handles lazy installation of heavy dependencies at runtime."""

    def __init__(self):
        # In PyInstaller exe, __file__ might not be available, so use cwd
        try:
            # Try to get the directory of the current script
            current_dir = Path(__file__).parent
        except (NameError, AttributeError):
            # In exe or when __file__ is not available, use current working directory
            current_dir = Path.cwd()
        print(f"🔧 Current directory for deps: {current_dir}")
        self.deps_dir = current_dir / "deps"
        self.installed_packages = set()
        self._ensure_deps_dir()

        # Debug info for exe environment
        print(f"🔧 Lazy installer initialized - deps directory: {self.deps_dir}")
        print(f"📂 Current working directory: {Path.cwd()}")

    def _ensure_deps_dir(self) -> None:
        """Ensure the deps directory exists."""
        self.deps_dir.mkdir(exist_ok=True)

    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed and importable."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            # In PyInstaller exe, some modules might not be available for import checking
            # even if they're installed at runtime. Check if they're in our deps directory.
            module_path = self.deps_dir / package_name
            if module_path.exists() or (module_path / "__init__.py").exists():
                return True

            # Also check if package files exist in deps (for packages without __init__.py)
            package_in_deps = False
            for file in self.deps_dir.rglob("*"):
                if (
                    package_name in file.name
                    or package_name.replace("-", "_") in file.name
                ):
                    package_in_deps = True
                    break

            return package_in_deps

    def _run_command(self, cmd: List[str], cwd: Optional[str] = None) -> bool:
        """Run a command and return success status with real-time output."""
        try:
            # Don't capture output - let it stream directly to console for real-time feedback
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=False,  # Don't raise exception on non-zero exit,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            # In PyInstaller exe, some commands might not be available or might fail
            print(
                f"⚠️ Warning: Failed to run command '{' '.join(cmd)}' - may not be available in exe environment"
            )
            return False

    def _has_uv(self) -> bool:
        """Check if uv package manager is available."""
        return self._run_command(["uv", "--version"])

    def _install_with_uv(
        self, packages: List[str], index_url: Optional[str] = None
    ) -> bool:
        """Install packages using uv."""
        cmd = ["uv", "pip", "install"] + packages
        if index_url:
            cmd.extend(["--index-url", index_url])

        # Use target directory for caching
        # cmd.extend(["--target", str(self.deps_dir)])

        print(f"⚠️ Using uv for installation: {' '.join(cmd)}")

        return self._run_command(cmd)

    def _install_with_pip(
        self, packages: List[str], index_url: Optional[str] = None
    ) -> bool:
        """Install packages using pip."""
        # cmd = ["python", "-m", "pip", "install", "--target", str(self.deps_dir)]

        if index_url:
            cmd.extend(["--index-url", index_url])

        cmd.extend(packages)
        print(f"⚠️ Falling back to pip for installation: {' '.join(cmd)}")

        return self._run_command(cmd)

    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version from nvidia-smi."""
        # Use nvidia-smi to detect CUDA version
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                output = result.stdout
                # Look for CUDA version in nvidia-smi output
                for line in output.split("\n"):
                    if "CUDA Version:" in line:
                        version = line.split("CUDA Version:")[1].strip()  # type: ignore
                        parts = version.split(".")
                        if len(parts) >= 2:
                            return f"cu{parts[0]}{parts[1]}"
        except:
            pass

        return None

    def _detect_rocm(self) -> bool:
        """Detect if ROCm is available."""
        try:
            # Try to run rocm-smi
            result = subprocess.run(["rocm-smi"], capture_output=True, check=False)
            return result.returncode == 0
        except:
            return False

    def _detect_environment(self) -> Tuple[str, Optional[str]]:
        """Detect the current environment (CPU, CUDA, ROCm)."""
        # Check for ROCm first
        if self._detect_rocm():
            return "rocm", None

        # Check for CUDA
        cuda_version = self._detect_cuda_version()
        if cuda_version:
            return "cuda", cuda_version

        # Default to CPU
        return "cpu", None

    def _get_package_variants(
        self, environment: str, cuda_version: Optional[str]
    ) -> dict:
        """Get the appropriate package variants for the detected environment."""
        variants = {}

        # Default packages that work on any environment (from pyproject.toml)
        variants.update(
            {
                # ML/Audio processing packages
                "whisper": {
                    "packages": ["openai-whisper>=20250625"],
                    "index_url": None,
                },
                "coqui_tts": {
                    "packages": ["coqui-tts>=0.27.1"],
                    "index_url": None,
                },
                "f5_tts": {
                    "packages": ["f5-tts>=1.1.9"],
                    "index_url": None,
                },
                "pyannote_audio": {
                    "packages": ["pyannote-audio>=3.3.2"],
                    "index_url": None,
                },
                "transformers": {
                    "packages": ["transformers>=4.56.0"],
                    "index_url": None,
                },
                "vocos": {
                    "packages": ["vocos>=0.1.0"],
                    "index_url": None,
                },
                "silero_vad": {
                    "packages": ["silero-vad>=6.0.0"],
                    "index_url": None,
                },
                "huggingface_hub": {
                    "packages": ["huggingface-hub[hf-xet]>=0.34.4"],
                    "index_url": None,
                },
                "edge_tts": {
                    "packages": ["edge-tts>=7.2.3"],
                    "index_url": None,
                },
                # Utility packages for smaller binary size
                "ffmpeg_python": {
                    "packages": ["ffmpeg-python>=0.2.0"],
                    "index_url": None,
                },
                "openai": {
                    "packages": ["openai>=1.102.0"],
                    "index_url": None,
                },
                "pydub": {
                    "packages": ["pydub>=0.25.1"],
                    "index_url": None,
                },
                "soundfile": {
                    "packages": ["soundfile>=0.13.1"],
                    "index_url": None,
                },
                "tqdm": {
                    "packages": ["tqdm>=4.65.0"],
                    "index_url": None,
                },
            }
        )

        # Environment-specific overrides
        if environment == "cpu":
            variants["torch"] = {
                "packages": ["torch", "torchvision", "torchaudio"],
                "index_url": "https://download.pytorch.org/whl/cpu",
            }
            variants["audio_separator"] = {
                "packages": ["audio-separator"],
                "index_url": None,
            }
        elif environment == "cuda":
            if cuda_version in ["cu126", "cu128", "cu129"]:
                variants["torch"] = {
                    "packages": ["torch", "torchvision", "torchaudio"],
                    "index_url": f"https://download.pytorch.org/whl/{cuda_version}",
                }
                variants["audio_separator"] = {
                    "packages": ["audio-separator[gpu]"],
                    "index_url": None,
                }
            else:
                # Fallback to latest CUDA version
                variants["torch"] = {
                    "packages": ["torch", "torchvision", "torchaudio"],
                    "index_url": "https://download.pytorch.org/whl/cu129",
                }
                variants["audio_separator"] = {
                    "packages": ["audio-separator[gpu]"],
                    "index_url": None,
                }
        elif environment == "rocm":
            variants["torch"] = {
                "packages": ["torch", "torchvision", "torchaudio"],
                "index_url": "https://download.pytorch.org/whl/rocm6.0",
            }
            variants["audio_separator"] = {
                "packages": ["audio-separator"],
                "index_url": None,
            }

        return variants

    def _install_package_group(self, package_name: str, config: dict) -> bool:
        """Install a group of packages (e.g., torch variants)."""
        packages = config["packages"]
        index_url = config["index_url"]

        print(f"📦 Installing {package_name}: {' '.join(packages)}")

        # Try uv first, fallback to pip
        if self._has_uv():
            success = self._install_with_uv(packages, index_url)
            if not success:
                print(f"⚠️ uv installation failed, falling back to pip...")
                success = self._install_with_pip(packages, index_url)
        else:
            success = self._install_with_pip(packages, index_url)

        if success:
            self.installed_packages.add(package_name)
            print(f"✅ Successfully installed {package_name}")
        else:
            print(f"❌ Failed to install {package_name}")

        return success

    def ensure_torch(self) -> bool:
        """Ensure torch is installed with the correct variant."""
        if self._is_package_installed("torch"):
            return True

        environment, cuda_version = self._detect_environment()
        variants = self._get_package_variants(environment, cuda_version)

        if "torch" not in variants:
            print("❌ No suitable torch variant found for this environment")
            return False

        return self._install_package_group("torch", variants["torch"])

    def ensure_audio_separator(self) -> bool:
        """Ensure audio-separator is installed with the correct variant."""
        if self._is_package_installed("audio_separator"):
            return True

        environment, cuda_version = self._detect_environment()
        variants = self._get_package_variants(environment, cuda_version)

        if "audio_separator" not in variants:
            print("❌ No suitable audio-separator variant found for this environment")
            return False

        return self._install_package_group(
            "audio-separator", variants["audio_separator"]
        )

    def _install_additional_package(self, package_name: str) -> bool:
        """Install an additional package if not already installed."""
        # Convert package names to module names for checking
        module_name_map = {
            "whisper": "whisper",
            "coqui_tts": "TTS",
            "f5_tts": "f5_tts",
            "pyannote_audio": "pyannote",
            "transformers": "transformers",
            "vocos": "vocos",
            "silero_vad": "silero_vad",
            "huggingface_hub": "huggingface_hub",
            "edge_tts": "edge_tts",
            "ffmpeg_python": "ffmpeg",
            "openai": "openai",
            "pydub": "pydub",
            "soundfile": "soundfile",
            "tqdm": "tqdm",
        }

        module_name = module_name_map.get(package_name, package_name)
        if self._is_package_installed(module_name):
            return True

        environment, cuda_version = self._detect_environment()
        variants = self._get_package_variants(environment, cuda_version)

        if package_name not in variants:
            print(f"⚠️ No variant found for {package_name} in environment {environment}")
            return False

        return self._install_package_group(package_name, variants[package_name])

    def ensure_dependencies(self) -> bool:
        """Ensure all required heavy dependencies are installed."""
        print("🔍 Checking for required dependencies...")

        # Add deps directory to Python path
        if str(self.deps_dir) not in sys.path:
            sys.path.insert(0, str(self.deps_dir))
            print(f"📂 Added {self.deps_dir} to Python path")

        success = True

        # Ensure torch
        if not self.ensure_torch():
            success = False

        # Ensure audio-separator
        if not self.ensure_audio_separator():
            success = False

        # Ensure additional packages
        additional_packages = [
            "whisper",
            "coqui_tts",
            "f5_tts",
            "pyannote_audio",
            "transformers",
            "vocos",
            "silero_vad",
            "huggingface_hub",
            "edge_tts",
            "ffmpeg_python",
            "openai",
            "pydub",
            "soundfile",
            "tqdm",
        ]
        for package in additional_packages:
            if not self._install_additional_package(package):
                success = False

        if success:
            print("✅ All dependencies installed successfully")
        else:
            print("❌ Some dependencies failed to install")
            print(
                "⚠️ Warning: The application may not function correctly without all dependencies."
            )
            print(
                "   This can happen in restricted environments where package installation is not allowed."
            )

        return success

    def clear_cache(self) -> None:
        """Clear the dependency cache."""
        if self.deps_dir.exists():
            shutil.rmtree(self.deps_dir)
            self.deps_dir.mkdir()
            print(f"🗑️ Cleared dependency cache in {self.deps_dir}")


# Global instance for easy access
_installer = None


def get_installer() -> LazyInstaller:
    """Get the global lazy installer instance."""
    global _installer
    if _installer is None:
        _installer = LazyInstaller()
    return _installer


def ensure_dependencies() -> bool:
    """Ensure all required dependencies are installed."""
    return get_installer().ensure_dependencies()


def clear_cache() -> None:
    """Clear the dependency cache."""
    get_installer().clear_cache()


# Auto-ensure dependencies when module is imported
# if __name__ != "__main__":
#     ensure_dependencies()

def install_and_cleanup():
    """Install dependencies and clean up cache after pipeline completion."""
    installer = get_installer()
    installer.ensure_dependencies()
    # Cache cleanup would be called after pipeline completion
    # e.g., in main.py after all stages
