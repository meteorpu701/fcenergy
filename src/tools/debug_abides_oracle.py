# src/debug_abides_oracle.py
from abides_core import abides
from abides_markets.configs import rmsc04

def find_exchange(agents):
    for a in agents:
        if a.__class__.__name__ == "ExchangeAgent":
            return a
    return None

def main():
    date = "2025-12-29"
    cfg = rmsc04.build_config(date=date, seed=1)
    end_state = abides.run(cfg)

    agents = end_state.get("agents", [])
    ex = find_exchange(agents)
    if ex is None:
        print("[ERROR] No ExchangeAgent found")
        return

    oracle = getattr(ex, "oracle", None)
    print("exchange.oracle:", type(oracle))

    # print any attributes that look like mean/fundamental
    for attr in ["r_bar", "fundamental_mean", "mean", "mean_fundamental", "kappa", "fund_vol"]:
        if oracle is not None and hasattr(oracle, attr):
            try:
                print(f"oracle.{attr} =", getattr(oracle, attr))
            except Exception as e:
                print(f"oracle.{attr} -> error {e}")

    # also print custom_properties if present in cfg
    print("custom_properties keys:", list(cfg.get("custom_properties", {}).keys()))

if __name__ == "__main__":
    main()