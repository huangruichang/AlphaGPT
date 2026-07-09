"""
A-share Alpha Mining Pipeline — Master Run Script

Data source: Tushare (primary, requires TUSHARE_TOKEN in .env)
             Akshare (fallback if tushare is unavailable)

Usage:
    python run_ashare.py --step db        # Create database tables
    python run_ashare.py --step download  # Download A-share data via tushare/akshare
    python run_ashare.py --step train     # Run factor mining (engine.py)
    python run_ashare.py --step all       # Run all steps in sequence

Examples:
    python run_ashare.py --step db
    python run_ashare.py --step download --all   # download top-500 by market cap
    python run_ashare.py --step download --days 365
    python run_ashare.py --step all --days 1095

Setup:
    1. Register at https://tushare.pro/register (free)
    2. Get your token at https://tushare.pro/user/token
    3. Add to .env: TUSHARE_TOKEN=your_token_here
"""

import argparse
import subprocess
import sys
import os

# Ensure the project root is on sys.path so imports work correctly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_db():
    """Create database tables (stock_basic, daily)."""
    print("=" * 60)
    print("[Step 1/3] Initialising database schema …")
    print("=" * 60)
    try:
        from data_pipeline.ashare_db import init_db
        init_db()
        print("[OK] Database tables are ready.\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialise database: {e}")
        print("\nTroubleshooting:")
        print("  - Check that PostgreSQL is running")
        print("  - Verify DB credentials in .env (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)")
        print("  - Ensure the target database exists")
        sys.exit(1)


def run_download(use_index: bool, days_back: int):
    """Download A-share data via akshare and write to PostgreSQL."""
    print("=" * 60)
    print("[Step 2/3] Downloading A-share data …")
    print("=" * 60)
    try:
        from data_pipeline.ashare_download import main as download_main
        download_main(use_index=use_index, days_back=days_back)
        print("[OK] Download complete.\n")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check network connectivity (akshare needs internet access)")
        print("  - Verify PostgreSQL is running and accessible")
        print("  - Check that the stock pool APIs are available (HS300/ZZ500)")
        sys.exit(1)


def run_train():
    """Run the AlphaGPT factor mining engine."""
    print("=" * 60)
    print("[Step 3/3] Starting Alpha Mining (AlphaGPT) …")
    print("=" * 60)
    try:
        # Run engine as a module so relative imports work correctly
        result = subprocess.run(
            [sys.executable, "-m", "model_core.engine"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        print("[OK] Training complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with exit code {e.returncode}.")
        print("\nTroubleshooting:")
        print("  - Check that data has been downloaded first (--step download)")
        print("  - Verify model_core/config.py paths point to valid data")
        print("  - Ensure PyTorch is installed and a GPU is available (optional)")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="A-share Alpha Mining Pipeline — master run script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ashare.py --step db              Create database tables
  python run_ashare.py --step download         Download HS300+ZZ500 data (default 2 years)
  python run_ashare.py --step download --all   Download top-500 by market cap
  python run_ashare.py --step download --days 365
  python run_ashare.py --step train            Run AlphaGPT factor mining
  python run_ashare.py --step all              Run full pipeline (db → download → train)
  python run_ashare.py --step all --days 1095  Full pipeline with 3 years of data
        """,
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["db", "download", "train", "all"],
        help="Pipeline step to run: db | download | train | all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="use_all",
        help="Use top-500-by-market-cap instead of HS300+ZZ500 (download step only)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Days of history to download (default: 730 = 2 years)",
    )
    args = parser.parse_args()

    step = args.step
    use_index = not args.use_all   # --all flag means use_index=False

    if step == "db":
        run_db()

    elif step == "download":
        run_db()   # ensure tables exist before downloading
        run_download(use_index=use_index, days_back=args.days)

    elif step == "train":
        run_train()

    elif step == "all":
        print("\nRunning full A-share pipeline …\n")
        run_db()
        run_download(use_index=use_index, days_back=args.days)
        run_train()
        print("=" * 60)
        print("[DONE] Full pipeline completed successfully!")
        print("=" * 60)

    print("\nResult files generated:")
    print("  - best_ashare_strategy.json  (best discovered factor formula)")
    print("  - training_history.json      (training metrics)\n")


if __name__ == "__main__":
    main()
