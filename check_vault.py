import json, os, pathlib

# Check what aegis sees as its config
aegis_config = pathlib.Path("/workspace/.hermes-aegis/config.json")
if aegis_config.exists():
    with open(aegis_config) as f:
        cfg = json.load(f)
    print("=== Aegis config ===")
    vault = cfg.get("vault", {})
    for k, v in vault.items():
        v = str(v)
        print(f"  {k}: {v[:20]}...{v[-5:]} (len={len(v)})")
else:
    print("No aegis config.json at /workspace/.hermes-aegis/")

# Check hermes auth.json
auth_paths = [
    pathlib.Path.home() / ".hermes" / "auth.json",
    pathlib.Path("/workspace/.hermes/auth.json"),
]
for p in auth_paths:
    if p.exists():
        print(f"\n=== {p} ===")
        with open(p) as f:
            auth = json.load(f)
        providers = auth.get("providers", {})
        for name, prov in providers.items():
            agent_key = prov.get("agent_key", "")
            access_token = prov.get("access_token", "")
            if agent_key:
                print(f"  {name}.agent_key: {agent_key[:20]}... (len={len(agent_key)})")
            if access_token:
                print(f"  {name}.access_token: {access_token[:20]}... (len={len(access_token)})")
    else:
        print(f"Not found: {p}")

# Check vault keyring store
vault_dir = pathlib.Path("/workspace/.hermes-aegis/vault")
if vault_dir.exists():
    print(f"\n=== Vault dir: {vault_dir} ===")
    for f in vault_dir.iterdir():
        print(f"  {f.name}: {f.stat().st_size} bytes")
else:
    print(f"\nNo vault dir at {vault_dir}")

