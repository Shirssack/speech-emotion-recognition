print("Testing utils.py...")

try:
    from utils import extract_feature, get_feature_vector_length
    print("[OK] utils.py imported successfully")

    length = get_feature_vector_length()
    print(f"[OK] Feature vector length: {length}")

except Exception as e:
    print(f"[FAILED] Error: {e}")
    import traceback
    traceback.print_exc()
