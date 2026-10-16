import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def run(*command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Install a pinned CPU-only NeMo Gym runtime without sudo")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python-install-dir", type=Path)
    parser.add_argument("--resolve-lock", type=Path, help="Resolve dependencies for this Python/platform, then exit")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((HERE / "sources.lock.json").read_text())
    if not root.exists():
        run("git", "clone", "--filter=blob:none", "https://github.com/NVIDIA-NeMo/Gym", root)
        run("git", "-C", root, "checkout", lock["nemo_gym_commit"])
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["nemo_gym_commit"]:
        raise ValueError("Choose a fresh --root; existing NeMo Gym checkout has a different revision")
    patches = sorted((HERE.parent / "patches").glob("*.patch"))
    for patch in patches:
        applied = subprocess.run(
            ["git", "-C", str(root), "apply", "--reverse", "--check", str(patch.resolve())], capture_output=True
        )
        if applied.returncode:
            run("git", "-C", root, "apply", "--check", patch.resolve())
            run("git", "-C", root, "apply", patch.resolve())
    (root / "cache").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(root / "cache" / "uv"),
        "NLTK_DATA": str(root / "nltk_data"),
        "NEMO_GYM_TAU2_BENCH_REF": lock["tau_commit"],
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    if args.python_install_dir:
        env["UV_PYTHON_INSTALL_DIR"] = str(args.python_install_dir.resolve())
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        run("uv", "venv", "--managed-python", "--python", lock["python"], root / ".venv", env=env)
    env["PATH"] = f"{python.parent}:{env['PATH']}"
    requirements = [
        f"git+https://github.com/abukharin-nv/verifiable-instructions@{lock['verifiable_instructions_commit']}",
        f"tau2[knowledge] @ git+https://github.com/bxyu-nvidia/tau2-bench@{lock['tau_commit']}",
        "reasoning-gym==0.1.25",
        "rank-bm25",
        "audioop-lts",
        "spacy",
        "emoji",
        "syllapy",
        "unicodedata2",
        "rich",
        "numpy==2.2.6",
        "math-verify==0.8.0",
        "openapi-schema-validator==0.6.3",
        "xmltodict==1.0.2",
        "tomli-w==1.2.0",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    ]
    if args.resolve_lock:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in") as source:
            source.write("\n".join(requirements) + "\n")
            source.flush()
            run(
                "uv",
                "pip",
                "compile",
                "--python",
                python,
                "--no-header",
                "--no-annotate",
                "--output-file",
                args.resolve_lock.resolve(),
                root / "pyproject.toml",
                source.name,
                env=env,
            )
        return
    requirements_lock = HERE / "requirements.lock.txt"
    run("uv", "pip", "sync", "--python", python, requirements_lock, env=env)
    run("uv", "pip", "install", "--python", python, "--no-deps", "-e", root, cwd=root, env=env)
    run(
        python,
        "-m",
        "nltk.downloader",
        "-d",
        root / "nltk_data",
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger_eng",
        cwd=root,
        env=env,
    )
    # Native benchmark preparation owns task schema and the no-LLM-judge policy.
    run(
        python,
        "-m",
        "benchmarks.tau2.prepare",
        "tau2",
        cwd=root,
        env={**env, "NEMO_GYM_TAU2_BENCH_DATA_REF": lock["tau_data_commit"]},
    )
    run(
        python,
        "-c",
        "from resources_servers.ifbench.setup_ifbench import ensure_ifbench; ensure_ifbench()",
        cwd=root,
        env=env,
    )
    run(
        python,
        "-c",
        "from pathlib import Path; from responses_api_agents.tau2.source import ensure_tau2_data_dir; ensure_tau2_data_dir(Path('responses_api_agents/tau2/tau2_data'))",
        cwd=root,
        env=env,
    )
    patched_paths = {
        line.removeprefix("+++ b/")
        for patch in patches
        for line in patch.read_text().splitlines()
        if line.startswith("+++ b/")
    }
    patched = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in patched_paths}
    packages = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True, env=env)
    assets = {}
    for directory in (
        root / "nltk_data",
        root / "responses_api_agents/tau2/tau2_data",
        root / "resources_servers/ifbench/.ifbench",
        root / "benchmarks/tau2/data",
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not {".git", "__pycache__"}.intersection(path.parts):
                with path.open("rb") as stream:
                    assets[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "sources": lock,
        "patched_sha256": patched,
        "assets_sha256": assets,
        "patches_sha256": {patch.name: hashlib.sha256(patch.read_bytes()).hexdigest() for patch in patches},
        "requirements_sha256": hashlib.sha256(requirements_lock.read_bytes()).hexdigest(),
        "packages": packages.splitlines(),
    }
    (root / "miles-runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"NeMo Gym CPU runtime ready: {python}")


if __name__ == "__main__":
    main()
