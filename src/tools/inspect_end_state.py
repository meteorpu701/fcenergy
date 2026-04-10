from src.sim.abides_simulation import run_rmsc04_simulation

if __name__ == "__main__":
    end_state = run_rmsc04_simulation("2020-06-05")
    print("end_state keys:", list(end_state.keys()))

    agents = end_state.get("agents")
    print("agents type:", type(agents))

    if isinstance(agents, list):
        print(f"Number of agents: {len(agents)}")

        # Inspect first few agents
        for idx, agent in enumerate(agents[:5]):
            print(f"\n=== Agent {idx} ===")
            print("type:", type(agent))
            print("class name:", agent.__class__.__name__)
            attrs = [a for a in dir(agent) if not a.startswith("_")]
            print("attrs (first 40):", attrs[:40])
    else:
        print("agents is not a list, got:", type(agents))