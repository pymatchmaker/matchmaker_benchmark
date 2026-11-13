"""Install the same immutable Matchmaker sources in every CI job.

Pending changes are applied only in a temporary checkout, without creating
commits or changing upstream branches. Remove a patch once it is in BASE.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY = "https://github.com/pymatchmaker/matchmaker.git"
# develop with #79, #80 and #81: DPDannenberg, its initial-window fix, 3 kHz SPARC.
BASE = "25960ea95aadf1b98d1bc753c8be133e0bc859ff"
PATCHES = (
    # Multi-Fold Arzt: retain the bounded repeat enumeration used by diagnostics.
    ("678366b034edc7cb10e3cf71bb1755b1a679e6c1",
     "0e673e2bfebd86c1e814ed9f5270b926c053a0ae"),
)


def main():
    with tempfile.TemporaryDirectory(prefix="benchmark-matchmaker-") as folder:
        source = Path(folder)

        def git(*args, **kwargs):
            return subprocess.run(
                ["git", "-C", str(source), *args], check=True, **kwargs
            )

        git("init", "--quiet")
        git("config", "core.autocrlf", "false")
        git("remote", "add", "origin", REPOSITORY)
        refs = sorted({BASE, *(ref for pair in PATCHES for ref in pair)})
        git("fetch", "--quiet", "--depth=1", "origin", *refs)
        git("checkout", "--quiet", "--detach", BASE)
        patches = []
        for base, head in PATCHES:
            patch = git("diff", "--binary", base, head, capture_output=True).stdout
            git("apply", "--check", "-", input=patch)
            git("apply", "-", input=patch)
            patches.append({
                "base": base, "head": head,
                "sha256": hashlib.sha256(patch).hexdigest(),
            })
        provenance = {
            "matchmaker_base_commit": BASE,
            "matchmaker_patches": patches,
        }
        manifest = source / "matchmaker" / "assets" / "benchmark-source.json"
        manifest.write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance, indent=2), flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *sys.argv[1:], str(source)],
            check=True,
        )


if __name__ == "__main__":
    main()
