import sys
sys.path.insert(0, '.')
print("starting import...")
try:
    from cognitive_memory.store import CognitiveMemoryStore
    print("import ok")
except Exception as e:
    print(f"import failed: {e}")
