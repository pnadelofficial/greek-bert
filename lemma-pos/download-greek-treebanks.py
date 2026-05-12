import argparse
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCES = {
    "agdt": {
        "name": "AGDT 2.1 (PerseusDL)",
        "url": "https://github.com/PerseusDL/treebank_data.git",
        "sparse_paths": ["v2.1/Greek"],
        "xml_glob": "v2.1/Greek/texts/*.xml",
        "license": "CC BY-SA 3.0",
        "note": "Homer, Hesiod, Aeschylus, Sophocles, Plato, Thucydides, ~200k tokens",
    },
    "gorman": {
        "name": "Gorman Trees (vgorman1)",
        "url": "https://github.com/vgorman1/Greek-Dependency-Trees.git",
        "sparse_paths": ["xml versions"],   # XML files at root level
        "xml_glob": "xml versions/*.xml",
        "license": "CC BY-SA 4.0",
        "note": "Attic prose (Aeschines, Demosthenes, Herodotus, Thucydides...), 550k+ tokens",
    },
    "harrington": {
        "name": "Harrington Trees (perseids-publications)",
        "url": "https://github.com/perseids-publications/harrington-trees.git",
        "sparse_paths": ["public/xml"],
        "xml_glob": "public/xml/**/*.xml",
        "license": "CC BY-SA 4.0",
        "note": "Tufts commentary treebanks, jmhgreek extended tagset",
    },
    "pedalion": {
        "name": "Pedalion Trees (perseids-publications)",
        "url": "https://github.com/perseids-publications/pedalion-trees.git",
        "sparse_paths": ["public/xml"],
        "xml_glob": "public/xml/**/*.xml",
        "license": "CC BY-SA 4.0",
        "note": "Grammar example sentences, ~12k tokens",
    },
    "daphne": {
        "name": "Daphne Trees — Mambrini Greek Poetry (perseids-publications)",
        "url": "https://github.com/perseids-publications/daphne-trees.git",
        "sparse_paths": ["public/xml"],
        "xml_glob": "public/xml/**/*.xml",
        "license": "CC BY-SA 4.0",
        "note": "Ancient Greek poetry treebanks by Francesco Mambrini",
    },
}

def run(cmd: list, cwd: Path = None, desc: str = "") -> bool:
    """Run a subprocess command, streaming output. Returns True on success."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] Command failed (exit {result.returncode}): {desc}")
        return False
    return True

def sparse_clone(url: str, dest: Path, sparse_paths: list) -> bool:
    """
    Perform a treeless sparse clone that fetches only the specified subdirectories.
    Much faster than a full clone for large repos like treebank_data.
    """
    if dest.exists():
        print(f"  Directory already exists, skipping clone: {dest}")
        return True
 
    print(f"\n  Cloning {url}")
    print(f"  → {dest}")
 
    # Step 1: treeless clone with no checkout (fetches tree metadata only)
    ok = run([
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--no-checkout",
        "--sparse",
        url,
        str(dest),
    ], desc="sparse clone")
    if not ok:
        return False
 
    # Step 2: configure sparse checkout to only materialise wanted paths
    ok = run(
        ["git", "sparse-checkout", "set"] + sparse_paths,
        cwd=dest,
        desc="sparse-checkout set",
    )
    if not ok:
        return False
 
    # Step 3: actually check out the files
    ok = run(["git", "checkout"], cwd=dest, desc="checkout")
    return ok
 
def is_greek_xml(path: Path) -> bool:
    """
    Quick check: does this XML file contain Greek treebank data with postag attributes?
    We check the xml:lang attribute and/or sample the first <word> element.
    """
    try:
        # Only parse the first 4096 bytes to keep this fast
        with path.open('rb') as f:
            head = f.read(4096).decode('utf-8', errors='replace')
 
        # Must have word elements with postag
        if 'postag=' not in head:
            return False
 
        # Language check: accept grc (Ancient Greek) or unspecified
        # Reject if explicitly labelled as Latin
        if 'xml:lang="la"' in head or 'xml:lang="lat"' in head:
            return False
 
        return True
    except Exception:
        return False

def count_tokens(path: Path) -> int:
    """Count non-artificial word tokens in an XML file."""
    try:
        tree = ET.parse(path)
        count = 0
        for word in tree.iter('word'):
            if word.get('artificial'):
                continue
            if word.get('postag') and word.get('form'):
                count += 1
        return count
    except Exception:
        return 0

def download_source(key: str, source: dict, base_dir: Path) -> dict:
    """Download a single source. Returns a summary dict."""
    print(f"\n{'='*60}")
    print(f"Source: {source['name']}")
    print(f"License: {source['license']}")
    print(f"Note: {source['note']}")
    print(f"{'='*60}")
 
    clone_dir = base_dir / "repos" / key
    out_dir = base_dir / key
 
    # Clone
    ok = sparse_clone(source["url"], clone_dir, source["sparse_paths"])
    if not ok:
        print(f"  [SKIP] Clone failed for {key}")
        return {"source": key, "status": "failed", "files": 0, "tokens": 0}
 
    # Find XML files
    xml_files = sorted(clone_dir.glob(source["xml_glob"]))
    if not xml_files:
        # Try recursive glob if flat glob found nothing
        xml_files = sorted(clone_dir.rglob("*.xml"))
 
    # Filter to Greek files with postag
    greek_files = [f for f in xml_files if is_greek_xml(f)]
 
    if not greek_files:
        print(f"  [WARN] No Greek XML files found in {key}")
        return {"source": key, "status": "no_greek_files", "files": 0, "tokens": 0}
 
    # Copy to output directory, preserving source structure
    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    copied = 0
 
    for src_file in greek_files:
        # Flatten into output dir with source prefix to avoid name collisions
        dest_name = f"{key}__{src_file.name}"
        dest_file = out_dir / dest_name
 
        shutil.copy2(src_file, dest_file)
        tokens = count_tokens(dest_file)
        total_tokens += tokens
        copied += 1
        print(f"  + {src_file.name} ({tokens:,} tokens)")
 
    print(f"\n  ✓ {copied} files, {total_tokens:,} tokens → {out_dir}")
    return {
        "source": key,
        "name": source["name"],
        "status": "ok",
        "files": copied,
        "tokens": total_tokens,
        "output_dir": str(out_dir),
        "license": source["license"],
    }
 
 
def print_summary(results: list, out_dir: Path):
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    total_files = 0
    total_tokens = 0
    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        files = r.get("files", 0)
        tokens = r.get("tokens", 0)
        print(f"  {status}  {r['source']:15s}  {files:4d} files  {tokens:>10,} tokens")
        total_files += files
        total_tokens += tokens
    print(f"{'─'*60}")
    print(f"     {'TOTAL':15s}  {total_files:4d} files  {total_tokens:>10,} tokens")
    print(f"\nXML files are in: {out_dir}")
    print(f"  Each source has its own subdirectory.")
    print(f"\nNext step:")
    print(f"  python convert_to_tsv.py {out_dir}/*/*.xml -o greek_morphology.tsv")

def main():
    parser = argparse.ArgumentParser(
        description="Download Ancient Greek AGDT treebanks from GitHub."
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("./treebanks"),
        help="Output directory (default: ./treebanks)",
    )
    parser.add_argument(
        "--sources", nargs="+", choices=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        help="Which sources to download (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available sources and exit",
    )
    parser.add_argument(
        "--keep-repos", action="store_true",
        help="Keep cloned git repos after copying XML files (default: delete them)",
    )
    args = parser.parse_args()
 
    if args.list:
        print("Available sources:")
        for key, src in SOURCES.items():
            print(f"  {key:15s} — {src['name']}")
            print(f"  {'':15s}   {src['note']}")
            print(f"  {'':15s}   License: {src['license']}")
            print()
        return
 
    # Check git is available
    if not shutil.which("git"):
        print("[ERROR] git is not installed or not on PATH.")
        sys.exit(1)
 
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
 
    results = []
    for key in args.sources:
        source = SOURCES[key]
        result = download_source(key, source, out_dir)
        results.append(result)
 
    # Optionally clean up repos (they can be large)
    if not args.keep_repos:
        repos_dir = out_dir / "repos"
        if repos_dir.exists():
            print(f"\nCleaning up cloned repos ({repos_dir})...")
            shutil.rmtree(repos_dir)
 
    print_summary(results, out_dir)
 
 
if __name__ == "__main__":
    main()



