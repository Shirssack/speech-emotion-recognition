"""
generate_hindi_csv.py - Generate CSV files for Hindi emotion dataset
"""

from create_csv import write_hindi_csv

if __name__ == "__main__":
    print("Generating Hindi dataset CSV files...")

    count = write_hindi_csv(
        'data/hindi',
        emotions=['angry', 'happy', 'neutral', 'sad', 'fear', 'disgust'],
        train_name='data/csv/train_hindi.csv',
        test_name='data/csv/test_hindi.csv',
        train_size=0.8,
        verbose=1
    )

    print(f"\nTotal files processed: {count}")
    print("CSV files created:")
    print("  - data/csv/train_hindi.csv")
    print("  - data/csv/test_hindi.csv")
