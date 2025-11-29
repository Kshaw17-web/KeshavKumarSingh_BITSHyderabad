# check_extractor_signature.py
import inspect
try:
    from src.extractor.bill_extractor import extract_bill_data
except Exception:
    try:
        from extractor.bill_extractor import extract_bill_data
    except Exception as e:
        print("Failed to import extract_bill_data:", e)
        raise SystemExit(1)

print("Loaded from:", getattr(extract_bill_data, "__module__", "<unknown>"))
print("Callable?:", callable(extract_bill_data))
print("Signature:", inspect.signature(extract_bill_data))
try:
    import inspect
    src = inspect.getsource(extract_bill_data)
    print("\nFirst 40 lines of function:\n")
    print("\n".join(src.splitlines()[:40]))
except Exception as e:
    print("Could not get source:", e)
