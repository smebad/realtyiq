import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# Check that model and database exist before starting
def check_prerequisites() -> bool:
    issues = []

    if not (ROOT / "models" / "xgboost_price_model.pkl").exists():
        issues.append("Model not trained. Run: python -m src.ml.train")

    if not (ROOT / "realtyiq.db").exists():
        issues.append("Database empty. Run: python -m src.db.seed")

    if not (ROOT / "data" / "embeddings" / "faiss_index.bin").exists():
        issues.append("Search index missing. Run: python -m src.search.indexer")

    if issues:
        print("\n⚠️  Setup incomplete:")
        for issue in issues:
            print(f"   → {issue}")
        print()
        return False

    return True

# Run the full setup pipeline
def run_setup() -> None:
    print("🔧 Running setup pipeline...")

    steps = [
        ("Processing data",    [sys.executable, "scripts/process_data.py"]),
        ("Seeding database",   [sys.executable, "-m", "src.db.seed"]),
        ("Training model",     [sys.executable, "-m", "src.ml.train"]),
        ("Building search index", [sys.executable, "-m", "src.search.indexer"]),
    ]

    for name, cmd in steps:
        print(f"\n⏳ {name}...")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"❌ Failed at: {name}")
            sys.exit(1)
        print(f"✅ {name} complete")

# Start FastAPI and Streamlit processes
def start_api() -> subprocess.Popen:
    print("🚀 Starting FastAPI on http://localhost:8000")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--reload",
        "--port", "8000",
        "--host", "0.0.0.0",
    ], cwd=ROOT)

# Streamlit ui
def start_ui() -> subprocess.Popen:
    print("🎨 Starting Streamlit UI on http://localhost:8501")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit",
        "run", "ui/app.py",
        "--server.port", "8501",
    ], cwd=ROOT)

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="RealtyIQ launcher")
    parser.add_argument("--api-only",  action="store_true")
    parser.add_argument("--ui-only",   action="store_true")
    parser.add_argument("--setup",     action="store_true",
                        help="Run full setup pipeline before starting")
    args = parser.parse_args()

    print("\n🏠 RealtyIQ — AI Real Estate Platform")
    print("=" * 40)

    if args.setup:
        run_setup()

    if not check_prerequisites():
        answer = input("Run setup now? (y/n): ").strip().lower()
        if answer == "y":
            run_setup()
        else:
            print("Exiting. Run setup steps manually then retry.")
            sys.exit(1)

    processes = []

    try:
        if not args.ui_only:
            processes.append(start_api())
            time.sleep(3)

        if not args.api_only:
            processes.append(start_ui())

        print("\n✅ RealtyIQ is running!")
        print("   API:  http://localhost:8000")
        print("   Docs: http://localhost:8000/docs")
        print("   UI:   http://localhost:8501")
        print("\nPress Ctrl+C to stop.\n")

        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for p in processes:
            p.terminate()
        print("Goodbye.")


if __name__ == "__main__":
    main()