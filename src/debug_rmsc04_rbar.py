from abides_markets.configs import rmsc04

for d in ["2025-12-29", "2026-01-05"]:
    cfg = rmsc04.build_config(date=d, seed=1)
    print("\nDATE", d)
    print("keys:", list(cfg.keys()))
    print("oracle_params:", cfg.get("oracle_params"))