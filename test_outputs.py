import os

EXPECTED_FILES = [
    "output/ma_equity_curve.png",
    "output/mr_equity_curve.png",
    "output/ma_vs_mr_equity_curve.png",
    "output/cost_sensitive_analysis_equity_curves.png",
    "output/parameter_sensitivity_curves.png",
]

def check_outputs():
    print("Checking for expected output files...\n")
    all_good = True
    for path in EXPECTED_FILES:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            all_good = False
        elif os.path.getsize(path) == 0:
            print(f"  EMPTY (0 bytes): {path}")
            all_good = False
        else:
            size_kb = os.path.getsize(path) / 1024
            print(f"  OK: {path} ({size_kb:.1f} KB)")

    print()
    if all_good:
        print("All expected output files were created successfully.")
    else:
        print("Some files are missing or empty — check the errors above.")
    return all_good


if __name__ == "__main__":
    # Delete any old outputs first, so we know these are freshly created
    # by THIS run, not leftovers from a previous one
    for path in EXPECTED_FILES:
        if os.path.exists(path):
            os.remove(path)

    print("Old outputs cleared. Now running main.py...\n")
    import main
    main.main()

    print("\n" + "=" * 50 + "\n")
    check_outputs()