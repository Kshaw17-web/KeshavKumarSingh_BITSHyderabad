import importlib, inspect, sys

print("=== Python src.extractor Inspection ===")

try:
    m = importlib.import_module('src.extractor.bill_extractor')
    print("MODULE_LOADED_FROM:", getattr(m, '__file__', '<unknown>'))
    print("HAS_extract_bill_data:", hasattr(m, 'extract_bill_data'))
    if hasattr(m, 'extract_bill_data'):
        print("--- SOURCE SNIPPET (first 40 lines) ---")
        src = inspect.getsource(m.extract_bill_data).splitlines()
        for ln in src[:40]:
            print(ln)
except Exception as e:
    print("ERROR:", repr(e))
