import matplotlib.pyplot as plt
import pandas as pd

class Architecture:
    est_ram = 0
    est_flash = 0
    proxy_map50 = 0
    parameters = 0
    eval_id = ""

NAS_RESULTS_JSON = "nas_results.json"

def pareto_score(arch: Architecture,
                 ram_limit: int,
                 flash_limit: int,
                 ram_weight:   float = 0.3,
                 flash_weight: float = 0.1) -> float:
    """
    Combined score: high mAP AND low hardware cost.
    Both fractions normalised to their limits so they are
    comparable to mAP which is in [0, 1].
    """
    mAP        = arch.proxy_map50
    ram_frac   = arch.est_ram   / max(ram_limit,   1)
    flash_frac = arch.est_flash / max(flash_limit, 1)
    penalty    = ram_weight * ram_frac + flash_weight * flash_frac
    return mAP * (1.0 - penalty)


def is_dominated(a: Architecture,
                 population: list[Architecture]) -> bool:
    """True if any other candidate weakly dominates a on all objectives."""
    for b in population:
        if b.eval_id == a.eval_id:
            continue
        if (b.proxy_map50 >= a.proxy_map50 and
                b.est_ram   <= a.est_ram and
                b.est_flash <= a.est_flash):
            return True
    return False


def pareto_front(population: list[Architecture]) -> list[Architecture]:
    return [a for a in population if not is_dominated(a, population)]

def plot_nas_results(nas_results_json=NAS_RESULTS_JSON):
    # Load NAS results from JSON file
    df = pd.read_json(nas_results_json)

    archs = []
    for _, row in df.iterrows():
        arch = Architecture()
        arch.est_ram = row.get("est_ram", 0)
        arch.est_flash = row.get("est_flash", 0)
        arch.proxy_map50 = row.get("proxy_map50", 0)
        arch.parameters = row.get("parameters", 0)
        arch.eval_id = row.get("eval_id", "")
        archs.append(arch)
    
    #plot proxy mAP50 vs idx line plot. change line color based on generation
    plt.figure(figsize=(10, 6))
    for gen in sorted(df["generation"].dropna().unique()):
        gen_df = df[df["generation"] == gen]
        plt.plot(gen_df.index, gen_df["proxy_map50"], marker='o', label=f"Gen {int(gen)}")
    plt.title("NAS Results: Proxy mAP50 vs Index (color by generation)")
    plt.xlabel("Index")
    plt.ylabel("Proxy mAP50")
    plt.grid()
    plt.legend()
    plt.savefig("nas_results_plot.png")
    plt.close()
    

    # plot average proxy mAP50 per generation
    plt.figure(figsize=(10, 6))
    df["generation"] = df["generation"].fillna(-1).astype(int)
    gen_means = df.groupby("generation")["proxy_map50"].mean()
    plt.plot(gen_means.index, gen_means.values, marker='o')
    plt.title("Average Proxy mAP50 per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Average Proxy mAP50")
    plt.grid()
    plt.savefig("nas_results_gen_plot.png")
    plt.close()

    # plot params vs proxy mAP50 color coded by generation
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["parameters"], df["proxy_map50"], c=df["generation"], cmap="viridis", alpha=0.7)
    plt.title("NAS Results: Params vs Proxy mAP50 (color by generation)")
    plt.xlabel("Params")
    plt.ylabel("Proxy mAP50")
    plt.grid()
    plt.legend(*scatter.legend_elements(), title="Generation")
    plt.savefig("nas_results_params_plot.png")
    plt.close()

    # plot max proxy mAP50 per generation
    plt.figure(figsize=(10, 6))
    gen_max = df.groupby("generation")["proxy_map50"].max()
    plt.plot(gen_max.index, gen_max.values, marker='o')
    plt.title("Max Proxy mAP50 per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Max Proxy mAP50")
    plt.grid()
    plt.savefig("nas_results_gen_max_plot.png")
    plt.close()

    # plot pareto front of params vs proxy mAP50
    plt.figure(figsize=(10, 6))
    front = pareto_front(archs)
    plt.scatter(df["parameters"], df["proxy_map50"], c='lightgray', alpha=0.5, label="All Candidates")
    plt.scatter([a.parameters for a in front], [a.proxy_map50 for a in front], c='red', label="Pareto Front")
    plt.title("NAS Results: Pareto Front of Params vs Proxy mAP50")
    plt.xlabel("Params")
    plt.ylabel("Proxy mAP50")
    plt.grid()
    plt.legend()
    plt.savefig("nas_results_pareto_plot.png")
    plt.close()

    # plot pareto score vs generation
    plt.figure(figsize=(10, 6))
    df["pareto_score"] = [pareto_score(arch, 500_000, 600_000) for arch in archs]
    for gen in sorted(df["generation"].dropna().unique()):
        gen_df = df[df["generation"] == gen]
        plt.plot(gen_df.index, gen_df["pareto_score"], marker='o', label=f"Gen {int(gen)}")
    plt.title("NAS Results: Pareto Score vs Index (color by generation)")
    plt.xlabel("Index")
    plt.ylabel("Pareto Score")
    plt.grid()
    plt.legend()
    plt.savefig("nas_results_pareto_score_plot.png")
    plt.close()



if __name__ == "__main__":
    import time
    # every 3 minutes, re-plot the NAS results to see the progress
    while True:
        plot_nas_results()
        print("Plotted NAS results. Sleeping for 3 minutes...")
        time.sleep(180)