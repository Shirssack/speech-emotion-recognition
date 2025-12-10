#!/usr/bin/env python3
"""
validate_datasets.py - Dataset Validation Tool
Author: Shirssack

Validates that datasets are properly downloaded and structured.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


class DatasetValidator:
    """Validates dataset structure and provides helpful feedback."""

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.issues = []
        self.warnings = []
        self.info = []

    def validate_all(self):
        """Run all validation checks."""
        print("=" * 70)
        print("Dataset Validation Tool".center(70))
        print("=" * 70)
        print()

        # Check data folder exists
        if not os.path.exists(self.data_path):
            self.issues.append(f"Data folder '{self.data_path}' does not exist!")
            self._print_results()
            return False

        # Validate each dataset
        ravdess_ok = self.validate_ravdess()
        tess_ok = self.validate_tess()
        hindi_ok = self.validate_hindi()
        emodb_ok = self.validate_emodb()
        csv_ok = self.validate_csvs()

        # Print results
        self._print_results()

        # Overall status
        print("\n" + "=" * 70)
        if not self.issues:
            print("✓ VALIDATION PASSED".center(70))
            print("All datasets are properly set up!".center(70))
            return True
        else:
            print("✗ VALIDATION FAILED".center(70))
            print(f"Found {len(self.issues)} issue(s) that need attention".center(70))
            return False

    def validate_ravdess(self):
        """Validate RAVDESS dataset structure."""
        print("[1/5] Checking RAVDESS dataset...")

        ravdess_path = os.path.join(self.data_path, 'ravdess')

        if not os.path.exists(ravdess_path):
            self.issues.append(
                "RAVDESS dataset not found at 'data/ravdess/'\n"
                "       → Download from: https://zenodo.org/record/1188976\n"
                "       → See DATASET_SETUP_GUIDE.md for detailed instructions"
            )
            return False

        # Check for Actor folders
        actor_folders = [f for f in os.listdir(ravdess_path)
                        if f.startswith('Actor_') and os.path.isdir(os.path.join(ravdess_path, f))]

        if not actor_folders:
            self.issues.append(
                "RAVDESS dataset folder exists but no Actor_* folders found\n"
                "       → Expected: Actor_01 through Actor_24\n"
                "       → Check if files are in the correct location"
            )
            return False

        # Count WAV files
        wav_count = 0
        for actor in actor_folders:
            actor_path = os.path.join(ravdess_path, actor)
            wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            wav_count += len(wav_files)

        expected_actors = 24
        if len(actor_folders) < expected_actors:
            self.warnings.append(
                f"RAVDESS: Found {len(actor_folders)}/{expected_actors} actor folders\n"
                f"         Some actors may be missing"
            )

        self.info.append(f"RAVDESS: ✓ {len(actor_folders)} actors, {wav_count} audio files")
        return True

    def validate_tess(self):
        """Validate TESS dataset structure."""
        print("[2/5] Checking TESS dataset...")

        tess_path = os.path.join(self.data_path, 'tess')

        if not os.path.exists(tess_path):
            self.issues.append(
                "TESS dataset not found at 'data/tess/'\n"
                "       → Download from: https://doi.org/10.5683/SP2/E8H2MF\n"
                "       → See DATASET_SETUP_GUIDE.md for detailed instructions"
            )
            return False

        # Expected emotion folders
        expected_folders = [
            'OAF_angry', 'OAF_disgust', 'OAF_Fear', 'OAF_happy',
            'OAF_neutral', 'OAF_Pleasant_surprise', 'OAF_Sad',
            'YAF_angry', 'YAF_disgust', 'YAF_fear', 'YAF_happy',
            'YAF_neutral', 'YAF_pleasant_surprised', 'YAF_sad'
        ]

        found_folders = [f for f in os.listdir(tess_path)
                        if os.path.isdir(os.path.join(tess_path, f))]

        if not found_folders:
            self.issues.append(
                "TESS dataset folder exists but no emotion folders found\n"
                "       → Expected folders: OAF_angry, OAF_happy, YAF_angry, etc.\n"
                "       → Check if folders are extracted correctly"
            )
            return False

        # Count WAV files
        wav_count = 0
        for folder in found_folders:
            folder_path = os.path.join(tess_path, folder)
            if os.path.isdir(folder_path):
                wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                wav_count += len(wav_files)

        missing_folders = set(expected_folders) - set(found_folders)
        if missing_folders:
            self.warnings.append(
                f"TESS: Missing emotion folders: {', '.join(sorted(missing_folders))}"
            )

        self.info.append(f"TESS: ✓ {len(found_folders)} emotion folders, {wav_count} audio files")
        return True

    def validate_hindi(self):
        """Validate Hindi dataset structure."""
        print("[3/5] Checking Hindi dataset...")

        hindi_path = os.path.join(self.data_path, 'hindi')

        if not os.path.exists(hindi_path):
            self.warnings.append(
                "Hindi dataset not found (optional)\n"
                "         → The model works fine with just RAVDESS and TESS\n"
                "         → To use Hindi dataset, see DATASET_SETUP_GUIDE.md"
            )
            return False

        # Check for audio files (structure can vary)
        wav_count = 0
        for root, dirs, files in os.walk(hindi_path):
            wav_count += len([f for f in files if f.endswith('.wav')])

        if wav_count == 0:
            self.warnings.append(
                "Hindi dataset folder exists but no WAV files found\n"
                "         → Check folder structure in DATASET_SETUP_GUIDE.md"
            )
            return False

        self.info.append(f"Hindi: ✓ {wav_count} audio files")
        return True

    def validate_emodb(self):
        """Validate EmoDB dataset structure."""
        print("[4/5] Checking EmoDB dataset...")

        # Check both possible names
        emodb_path = os.path.join(self.data_path, 'emodb')
        emodb_hyphen_path = os.path.join(self.data_path, 'emo-db')

        if os.path.exists(emodb_hyphen_path) and not os.path.exists(emodb_path):
            self.warnings.append(
                "EmoDB dataset found as 'emo-db/' but code expects 'emodb/'\n"
                f"         → Rename folder: mv data/emo-db data/emodb\n"
                "         → Or the dataset will be ignored during training"
            )
            emodb_path = emodb_hyphen_path

        if not os.path.exists(emodb_path):
            self.warnings.append(
                "EmoDB dataset not found (optional)\n"
                "         → This is a German emotion dataset\n"
                "         → Not required for English/Hindi models"
            )
            return False

        # Count WAV files
        wav_count = 0
        for root, dirs, files in os.walk(emodb_path):
            wav_count += len([f for f in files if f.endswith('.wav')])

        if wav_count == 0:
            self.warnings.append(
                "EmoDB folder exists but no WAV files found"
            )
            return False

        self.info.append(f"EmoDB: ✓ {wav_count} audio files")
        return True

    def validate_csvs(self):
        """Validate CSV files."""
        print("[5/5] Checking CSV files...")

        csv_path = os.path.join(self.data_path, 'csv')

        if not os.path.exists(csv_path):
            self.warnings.append(
                "CSV folder not found\n"
                "         → Will be created automatically during training\n"
                "         → Or run: python generate_4class_csv.py"
            )
            return False

        # Check for expected CSV files
        expected_csvs = [
            'train_ravdess_4class.csv', 'test_ravdess_4class.csv',
            'train_tess_4class.csv', 'test_tess_4class.csv'
        ]

        found_csvs = [f for f in os.listdir(csv_path) if f.endswith('.csv')]

        if not found_csvs:
            self.warnings.append(
                "No CSV files found\n"
                "         → Will be generated automatically on first training\n"
                "         → Or run: python generate_4class_csv.py"
            )
            return False

        # Validate CSV paths exist
        broken_csvs = []

        try:
            import pandas as pd

            for csv_file in found_csvs:
                csv_full_path = os.path.join(csv_path, csv_file)
                try:
                    df = pd.read_csv(csv_full_path)
                    if 'path' in df.columns:
                        # Check if referenced files exist
                        missing_count = 0
                        for audio_path in df['path'].head(10):  # Check first 10
                            if not os.path.exists(audio_path):
                                missing_count += 1

                        if missing_count > 0:
                            broken_csvs.append(csv_file)
                except Exception as e:
                    self.warnings.append(f"Could not read {csv_file}: {e}")

            if broken_csvs:
                self.warnings.append(
                    f"CSV files reference missing audio files: {', '.join(broken_csvs)}\n"
                    "         → Delete CSV folder: rm -rf data/csv\n"
                    "         → Regenerate: python generate_4class_csv.py"
                )

            self.info.append(f"CSV: ✓ {len(found_csvs)} CSV files")

        except ImportError:
            # Pandas not available, skip CSV content validation
            self.info.append(f"CSV: ✓ {len(found_csvs)} CSV files (content validation skipped)")

        return True

    def _print_results(self):
        """Print validation results."""
        print()
        print("=" * 70)
        print("Validation Results".center(70))
        print("=" * 70)

        # Print info
        if self.info:
            print("\n✓ FOUND:")
            for msg in self.info:
                print(f"  {msg}")

        # Print warnings
        if self.warnings:
            print("\n⚠ WARNINGS:")
            for i, msg in enumerate(self.warnings, 1):
                lines = msg.split('\n')
                print(f"  {i}. {lines[0]}")
                for line in lines[1:]:
                    print(f"     {line}")

        # Print issues
        if self.issues:
            print("\n✗ ISSUES:")
            for i, msg in enumerate(self.issues, 1):
                lines = msg.split('\n')
                print(f"  {i}. {lines[0]}")
                for line in lines[1:]:
                    print(f"     {line}")

    def get_download_instructions(self):
        """Print download instructions for missing datasets."""
        print("\n" + "=" * 70)
        print("Quick Setup Instructions".center(70))
        print("=" * 70)
        print("""
1. Download RAVDESS:
   → Visit: https://zenodo.org/record/1188976
   → Download: Audio_Speech_Actors_01-24.zip
   → Extract and move Actor_* folders to: data/ravdess/

2. Download TESS:
   → Visit: https://doi.org/10.5683/SP2/E8H2MF
   → Download all emotion ZIP files (OAF_*, YAF_*)
   → Extract and move folders to: data/tess/

3. Verify setup:
   → Run: python validate_datasets.py

4. Start training:
   → Run: python train_transformer.py --epochs 15

For detailed instructions, see: DATASET_SETUP_GUIDE.md
""")


def main():
    """Main validation function."""
    validator = DatasetValidator()

    success = validator.validate_all()

    if not success:
        print()
        validator.get_download_instructions()
        sys.exit(1)
    else:
        print()
        print("You're ready to start training!")
        print()
        print("Next steps:")
        print("  • Quick test: python run_test.py")
        print("  • Train transformer: python train_transformer.py --epochs 15")
        print("  • Train traditional ML: python train_traditional_ml.py")
        print()
        sys.exit(0)


if __name__ == '__main__':
    main()
