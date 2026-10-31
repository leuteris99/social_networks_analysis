"""Social Network Analysis assignment: time-aware centrality measures & link prediction.

Part I: split the Stack Overflow temporal network into N equal time periods,
build an undirected graph for every period, plot how |V| and |E| evolve and
plot the distribution of five centrality measures for every period.

Part II: for every pair of successive periods keep the nodes present in both,
restrict both edge sets to them, plot their sizes and compute five similarity
matrices over all pairs of common nodes.

Part III: use every similarity measure as a link predictor. Train the optimal
union of score intervals R*_X on the first period of each pair and rank the
measures by their training and testing accuracy.
"""
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg

DATA_PATH = "data/sx-stackoverflow.txt.gz"
MAX_ROWS = 200_000   # only the first rows are used, the full network is too large for these measures
N = 5                # number of time periods
BINS = 30            # histogram bins
BETWEENNESS_K = 500  # sampled source nodes for approximate betweenness
FIG_DIR = "figures"


def load_edges(path, nrows):
    """Read the (source_id, target_id, timestamp) triplets."""
    return pd.read_csv(path, sep=" ", names=["u", "v", "t"], nrows=nrows)


def time_boundaries(edges, n):
    """Return the time instances t_0, ..., t_N with t_j = t_min + j * dt (Eq. 3)."""
    return np.linspace(edges.t.min(), edges.t.max(), n + 1)


def period_graphs(edges, bounds):
    """Build the undirected graph G[t_{j-1}, t_j] of every period T_j (Eq. 4-6).

    Each period is [t_{j-1}, t_j), except the last one which also contains t_max.
    Repeated edges collapse into one and self-loops are dropped (simple graph).
    """
    n = len(bounds) - 1
    period = np.minimum(np.digitize(edges.t, bounds) - 1, n - 1)
    graphs = []
    for j in range(n):
        e = edges[period == j]
        g = nx.Graph()
        g.add_edges_from(zip(e.u, e.v))
        g.remove_edges_from(nx.selfloop_edges(g))
        graphs.append(g)
    return graphs


def centralities(g):
    """Compute the five centrality measures of Part I.4 for every node of g."""
    adjacency = nx.to_scipy_sparse_array(g, dtype=float)
    largest_eigenvalue = scipy.sparse.linalg.eigsh(adjacency, k=1, which="LA")[0][0]
    return {
        "Degree": nx.degree_centrality(g),
        "Closeness": nx.closeness_centrality(g),
        "Betweenness": nx.betweenness_centrality(g, k=min(BETWEENNESS_K, len(g)), seed=0),
        "Eigenvector": nx.eigenvector_centrality(g, max_iter=1000),
        # Katz converges only for alpha < 1 / largest eigenvalue
        "Katz": nx.katz_centrality(g, alpha=0.9 / largest_eigenvalue),
    }


def print_periods(bounds, graphs):
    dates = pd.to_datetime(bounds, unit="s")
    for j, g in enumerate(graphs, 1):
        print(f"T{j}: {dates[j - 1]} -> {dates[j]}  |V|={g.number_of_nodes()}  |E|={g.number_of_edges()}")


def plot_sizes(graphs):
    """Time evolution of |V[t_{j-1}, t_j]| and |E[t_{j-1}, t_j]| (Part I.3)."""
    j = np.arange(1, len(graphs) + 1)
    plt.figure()
    plt.plot(j, [g.number_of_nodes() for g in graphs], "o-", label="|V|")
    plt.plot(j, [g.number_of_edges() for g in graphs], "s-", label="|E|")
    plt.xticks(j, [f"T{i}" for i in j])
    plt.xlabel("time period")
    plt.ylabel("count")
    plt.title("Nodes and edges per time period")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/part1_sizes.png", dpi=150)
    plt.close()


def plot_centralities(j, cents):
    """Histograms of relative frequencies for every centrality measure of period T_j (Part I.4)."""
    fig, axes = plt.subplots(1, len(cents), figsize=(4 * len(cents), 4))
    for ax, (name, c) in zip(axes, cents.items()):
        values = np.fromiter(c.values(), float)
        ax.hist(values, bins=BINS, weights=np.full(len(values), 1 / len(values)))
        ax.set_yscale("log")  # the distributions are heavy-tailed
        ax.set_title(name)
        ax.set_xlabel("centrality")
        ax.set_ylabel("relative frequency")
    fig.suptitle(f"Centrality distributions in T{j}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/part1_centralities_T{j}.png", dpi=150)
    plt.close(fig)


def part1():
    edges = load_edges(DATA_PATH, MAX_ROWS)
    bounds = time_boundaries(edges, N)
    graphs = period_graphs(edges, bounds)
    print_periods(bounds, graphs)
    plot_sizes(graphs)
    for j, g in enumerate(graphs, 1):
        print(f"computing centralities for T{j}")
        plot_centralities(j, centralities(g))
    return graphs


def common_subgraphs(g1, g2):
    """Return V*[t_{j-1}, t_{j+1}] and the graphs of E*[t_{j-1}, t_j], E*[t_j, t_{j+1}] (Eq. 7-9).

    Both graphs keep every node of V*, even if it has no edge left after the restriction.
    """
    nodes = sorted(set(g1) & set(g2))
    return nodes, g1.subgraph(nodes), g2.subgraph(nodes)


def similarities(g, nodes):
    """Similarity matrices over all pairs of `nodes`, computed on graph g (Part II.2).

    Row/column i of every matrix corresponds to nodes[i]. Unreachable pairs get
    S_GD = -inf.
    """
    a = nx.to_scipy_sparse_array(g, nodelist=nodes, dtype=float)
    deg = a.sum(axis=1)
    cn = (a @ a).toarray()
    union = deg[:, None] + deg[None, :] - cn
    # a common neighbour z of two distinct nodes has |Γ(z)| >= 2, so log(|Γ(z)|) > 0
    inv_log_deg = np.zeros_like(deg)
    inv_log_deg[deg > 1] = 1 / np.log(deg[deg > 1])
    return {
        "GD": -scipy.sparse.csgraph.shortest_path(a, unweighted=True, directed=False),
        "CN": cn,
        "JC": np.divide(cn, union, out=np.zeros_like(cn), where=union > 0),
        "A": (a @ scipy.sparse.diags(inv_log_deg) @ a).toarray(),
        "PA": np.outer(deg, deg),
    }


def plot_common_sizes(sizes):
    """|V*[t_{j-1}, t_{j+1}]|, |E*[t_{j-1}, t_j]| and |E*[t_j, t_{j+1}]| per pair (T_j, T_{j+1}) (Part II.1)."""
    j = np.arange(1, len(sizes) + 1)
    plt.figure()
    plt.plot(j, [s[0] for s in sizes], "o-", label="|V*[t_{j-1}, t_{j+1}]|")
    plt.plot(j, [s[1] for s in sizes], "s-", label="|E*[t_{j-1}, t_j]|")
    plt.plot(j, [s[2] for s in sizes], "^-", label="|E*[t_j, t_{j+1}]|")
    plt.xticks(j, [f"(T{i},T{i + 1})" for i in j])
    plt.xlabel("pair of successive time periods")
    plt.ylabel("count")
    plt.title("Common nodes and restricted edges per pair of periods")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/part2_common_sizes.png", dpi=150)
    plt.close()


def part2(graphs):
    """Return, for every pair (T_j, T_{j+1}), V*, the training/testing graphs and the similarity matrices."""
    pairs = []
    for j, (g1, g2) in enumerate(zip(graphs, graphs[1:]), 1):
        nodes, train, test = common_subgraphs(g1, g2)
        print(f"(T{j},T{j + 1}): |V*|={len(nodes)}  |E*_train|={train.number_of_edges()}  "
              f"|E*_test|={test.number_of_edges()}")
        # scores come from the training period only, so they can be used to predict the next one
        pairs.append({"nodes": nodes, "train": train, "test": test, "scores": similarities(train, nodes)})
    plot_common_sizes([(len(p["nodes"]), p["train"].number_of_edges(), p["test"].number_of_edges())
                       for p in pairs])
    return pairs


def train_ranges(scores, labels):
    """Training algorithm: the range set R*_X of Eq. 19 that maximises ACC on `labels` (Eq. 20).

    With λ = |E| / |E0|, Eq. 14 simplifies to ACC = (TP + TN) / |E0|, i.e. plain accuracy.
    Every distinct score value is therefore decided independently: it belongs to R*_X
    exactly when more of its pairs are edges than non-edges. Runs of consecutive
    selected values are merged into the intervals [S_L, S_U].
    """
    values, inverse = np.unique(scores, return_inverse=True)
    pos = np.bincount(inverse, weights=labels, minlength=len(values))
    neg = np.bincount(inverse, minlength=len(values)) - pos
    keep = pos > neg
    ranges = []
    for i in np.flatnonzero(keep):
        if ranges and keep[i - 1]:
            ranges[-1][1] = values[i]
        else:
            ranges.append([values[i], values[i]])
    return ranges


def in_ranges(scores, ranges):
    """True for every score that falls inside one of the intervals of R_X (Eq. 13)."""
    lows = np.array([r[0] for r in ranges])
    highs = np.array([r[1] for r in ranges])
    k = np.searchsorted(lows, scores, side="right") - 1  # last interval starting at or before the score
    return (k >= 0) & (scores <= highs[k])


def accuracy(scores, labels, ranges):
    """ACC(R_X, E) of Eq. 14-17, where `labels` marks the pairs of E0 that belong to E."""
    predicted = in_ranges(scores, ranges)
    hits = np.sum(predicted & labels)
    lam = labels.sum() / len(labels)
    tpr = hits / labels.sum()
    tnr = 1 - (predicted.sum() - hits) / (len(labels) - labels.sum())
    return lam * tpr + (1 - lam) * tnr


def part3(pairs):
    """Train R*_X on E*[t_{j-1}, t_j], test it on E*[t_j, t_{j+1}] and rank the measures (Part III)."""
    pd.set_option("display.float_format", "{:.6f}".format)
    for j, p in enumerate(pairs, 1):
        iu = np.triu_indices(len(p["nodes"]), 1)  # E0: every unordered pair u < v of V*
        train = nx.to_numpy_array(p["train"], nodelist=p["nodes"], dtype=bool)[iu]
        test = nx.to_numpy_array(p["test"], nodelist=p["nodes"], dtype=bool)[iu]
        rows = []
        for name, s in p["scores"].items():
            scores = s[iu]
            ranges = train_ranges(scores, train)
            rows.append({"measure": name, "intervals": len(ranges),
                         "train ACC": accuracy(scores, train, ranges),
                         "test ACC": accuracy(scores, test, ranges)})
        table = pd.DataFrame(rows).set_index("measure")
        print(f"\n(T{j},T{j + 1})  |E0|={len(train)}  |E_train|={train.sum()}  |E_test|={test.sum()}")
        print("training ranking:\n", table.sort_values("train ACC", ascending=False)[["intervals", "train ACC"]])
        print("testing ranking:\n", table.sort_values("test ACC", ascending=False)[["test ACC"]])


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    graphs = part1()
    pairs = part2(graphs)
    part3(pairs)
