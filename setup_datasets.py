#!/usr/bin/env python3
"""
setup_datasets.py - Interactive Dataset Setup Helper
Author: Shirssack

Interactive script to help users download and set up datasets.
"""

import os
import sys
import webbrowser
from pathlib import Path


class DatasetSetupHelper:
    """Interactive helper for setting up datasets."""

    def __init__(self):
        self.data_path = 'data'

    def run(self):
        """Run the interactive setup process."""
        self.print_banner()
        self.check_current_status()
        self.offer_help()

    def print_banner(self):
        """Print welcome banner."""
        print("=" * 70)
        print("Speech Emotion Recognition - Dataset Setup".center(70))
        print("=" * 70)
        print()
        print("This script will help you download and set up the required datasets.")
        print()

    def check_current_status(self):
        """Check which datasets are already set up."""
        print("Checking current status...")
        print()

        # Check data folder
        if not os.path.exists(self.data_path):
            print(f"✗ Data folder '{self.data_path}' does not exist")
            print(f"  Creating folder...")
            os.makedirs(self.data_path, exist_ok=True)
            print(f"  ✓ Created '{self.data_path}' folder")
            print()

        # Check RAVDESS
        ravdess_status = self.check_ravdess()
        print(f"{'✓' if ravdess_status else '✗'} RAVDESS: {'Ready' if ravdess_status else 'Not found'}")

        # Check TESS
        tess_status = self.check_tess()
        print(f"{'✓' if tess_status else '✗'} TESS: {'Ready' if tess_status else 'Not found'}")

        # Check Hindi (optional)
        hindi_status = self.check_hindi()
        print(f"{'✓' if hindi_status else '○'} Hindi: {'Ready' if hindi_status else 'Not found (optional)'}")

        print()

        if ravdess_status and tess_status:
            print("✓ All required datasets are ready!")
            print()
            print("You can now:")
            print("  • Run validation: python validate_datasets.py")
            print("  • Start training: python train_transformer.py --epochs 15")
            print()
            sys.exit(0)

    def check_ravdess(self):
        """Check if RAVDESS is set up."""
        ravdess_path = os.path.join(self.data_path, 'ravdess')
        if not os.path.exists(ravdess_path):
            return False

        actor_folders = [f for f in os.listdir(ravdess_path)
                        if f.startswith('Actor_') and
                        os.path.isdir(os.path.join(ravdess_path, f))]
        return len(actor_folders) > 0

    def check_tess(self):
        """Check if TESS is set up."""
        tess_path = os.path.join(self.data_path, 'tess')
        if not os.path.exists(tess_path):
            return False

        emotion_folders = [f for f in os.listdir(tess_path)
                          if os.path.isdir(os.path.join(tess_path, f)) and
                          ('OAF_' in f or 'YAF_' in f)]
        return len(emotion_folders) > 0

    def check_hindi(self):
        """Check if Hindi dataset is set up."""
        hindi_path = os.path.join(self.data_path, 'hindi')
        if not os.path.exists(hindi_path):
            return False

        # Check for any WAV files
        for root, dirs, files in os.walk(hindi_path):
            if any(f.endswith('.wav') for f in files):
                return True
        return False

    def offer_help(self):
        """Offer help options to the user."""
        print("=" * 70)
        print("What would you like to do?")
        print("=" * 70)
        print()
        print("1. Get download links for RAVDESS")
        print("2. Get download links for TESS")
        print("3. Get download links for Hindi dataset (optional)")
        print("4. View detailed setup instructions")
        print("5. Open validation tool")
        print("6. Exit")
        print()

        try:
            choice = input("Enter your choice (1-6): ").strip()

            if choice == '1':
                self.help_ravdess()
            elif choice == '2':
                self.help_tess()
            elif choice == '3':
                self.help_hindi()
            elif choice == '4':
                self.show_detailed_instructions()
            elif choice == '5':
                self.run_validation()
            elif choice == '6':
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("\nInvalid choice. Please try again.")
                self.offer_help()

        except KeyboardInterrupt:
            print("\n\nSetup cancelled.")
            sys.exit(0)

    def help_ravdess(self):
        """Help with RAVDESS dataset."""
        print("\n" + "=" * 70)
        print("RAVDESS Dataset Setup")
        print("=" * 70)
        print("""
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Download Link: https://zenodo.org/record/1188976

Steps:
1. Visit the link above (press Enter to open in browser)
2. Download: Audio_Speech_Actors_01-24.zip (~500MB)
3. Extract the ZIP file
4. Move all Actor_* folders to: data/ravdess/

Expected structure:
    data/ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...

After downloading, run: python validate_datasets.py
""")

        try:
            input("\nPress Enter to open download page in browser (or Ctrl+C to cancel)...")
            webbrowser.open('https://zenodo.org/record/1188976')
            print("✓ Opened in browser")
        except KeyboardInterrupt:
            print("\nCancelled")

        self.wait_and_continue()

    def help_tess(self):
        """Help with TESS dataset."""
        print("\n" + "=" * 70)
        print("TESS Dataset Setup")
        print("=" * 70)
        print("""
TESS (Toronto Emotional Speech Set)

Download Link: https://doi.org/10.5683/SP2/E8H2MF

Steps:
1. Visit the link above (press Enter to open in browser)
2. Download all emotion category ZIP files:
   - OAF_angry.zip, OAF_disgust.zip, OAF_Fear.zip, etc.
   - YAF_angry.zip, YAF_disgust.zip, YAF_fear.zip, etc.
3. Extract all ZIP files
4. Move all extracted folders to: data/tess/

Expected structure:
    data/tess/
    ├── OAF_angry/
    ├── OAF_disgust/
    ├── YAF_angry/
    └── ...

After downloading, run: python validate_datasets.py
""")

        try:
            input("\nPress Enter to open download page in browser (or Ctrl+C to cancel)...")
            webbrowser.open('https://doi.org/10.5683/SP2/E8H2MF')
            print("✓ Opened in browser")
        except KeyboardInterrupt:
            print("\nCancelled")

        self.wait_and_continue()

    def help_hindi(self):
        """Help with Hindi dataset."""
        print("\n" + "=" * 70)
        print("Hindi Dataset Setup")
        print("=" * 70)
        print("""
Hindi Emotion Speech Dataset (Optional)

Note: The Hindi dataset may require academic access or direct contact with
researchers. It is not publicly available like RAVDESS and TESS.

If you have access:
1. Place audio files in: data/hindi/my Dataset/
2. Organize by speaker and emotion
3. Run validation: python validate_datasets.py

The system works fine without the Hindi dataset using just RAVDESS and TESS.
""")

        self.wait_and_continue()

    def show_detailed_instructions(self):
        """Show detailed instructions."""
        guide_path = 'DATASET_SETUP_GUIDE.md'

        if os.path.exists(guide_path):
            print(f"\n✓ Opening {guide_path}...")
            print("\nPlease read the detailed guide for complete instructions.")
            print(f"Location: {os.path.abspath(guide_path)}")
        else:
            print(f"\n✗ {guide_path} not found in the current directory")
            print("\nPlease refer to the README.md or documentation.")

        self.wait_and_continue()

    def run_validation(self):
        """Run the validation tool."""
        print("\n" + "=" * 70)
        print("Running validation tool...")
        print("=" * 70)
        print()

        validation_script = 'validate_datasets.py'

        if os.path.exists(validation_script):
            import subprocess
            subprocess.run([sys.executable, validation_script])
        else:
            print(f"✗ {validation_script} not found")
            print("Please ensure you're in the project root directory")

        self.wait_and_continue()

    def wait_and_continue(self):
        """Wait for user to continue."""
        print()
        try:
            input("Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

        print()
        self.offer_help()


def main():
    """Main setup function."""
    try:
        helper = DatasetSetupHelper()
        helper.run()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled. Goodbye!")
        sys.exit(0)


if __name__ == '__main__':
    main()
