import traceback

print("=== ABIDES Installation Test ===")

try:
    from abides_core import abides
    print("[OK] Imported abides_core.abides")
except Exception as e:
    print("[ERROR] Failed to import abides_core.abides")
    traceback.print_exc()
    exit(1)

try:
    from abides_markets.configs import rmsc04
    print("[OK] Imported abides_markets.configs.rmsc04")
except Exception as e:
    print("[ERROR] Failed to import abides_markets.configs.rmsc04")
    traceback.print_exc()
    exit(1)

try:
    config = rmsc04.build_config(
        seed=123,
        date="20250101",  
        log_orders=False   
    )
    print("[OK] Config built successfully.")
except Exception as e:
    print("[ERROR] Failed to build RMSC04 config.")
    traceback.print_exc()
    exit(1)

try:
    print("[INFO] Running ABIDES simulation... (this may take a moment)")
    end_state = abides.run(config)
    print("[OK] ABIDES simulation completed.")
except Exception as e:
    print("[ERROR] ABIDES run failed.")
    traceback.print_exc()
    exit(1)

try:
    agents = end_state.get("agents", {})
    trades = end_state.get("trades", [])

    print(f"[INFO] Agents processed: {len(agents)}")
    print(f"[INFO] Trades generated: {len(trades)}")

    if trades:
        sample = trades[0]
        print("[INFO] Sample trade:")
        print(f"  Agent ID: {sample.get('agent_id', 'N/A')}")
        print(f"  Time:     {sample.get('timestamp', 'N/A')}")
        print(f"  Price:    {sample.get('price', 'N/A')}")

    print("\n=== SUCCESS: ABIDES is working correctly. ===")

except Exception as e:
    print("[ERROR] Could not inspect end_state output.")
    traceback.print_exc()
    exit(1)