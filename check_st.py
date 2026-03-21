import os, sys, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
sys.path.insert(0, '.')
print("loading sentence-transformers (offline)...", flush=True)
t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    print(f"  imported in {time.time()-t0:.1f}s", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/workspace/Projects/.huggingface_cache/hub')
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)
    emb = model.encode(["test sentence"])
    print(f"  encode done in {time.time()-t0:.1f}s, shape={emb.shape}", flush=True)
except Exception as e:
    print(f"failed: {e}")
    import traceback; traceback.print_exc()
print("done")
