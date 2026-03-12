from itertools import combinations
from math import ceil, log2

# ----------------------------
# 1. Storage model
# ----------------------------

F_rows = 1.8e9
S_F_GB = 41.85
factor = F_rows / (S_F_GB * 1e9)   # |F| / S(F)
beta = 0.10

domain = {
    "d_year": 7,
    "d_yearmonth": 84,
    "s_region": 5,
    "c_region": 5,
    "s_nation": 25,
    "c_nation": 25,
    "s_city": 250,
    "c_city": 250,
    "p_mfgr": 5,
    "p_category": 25,
    "p_brand1": 1000,
}

cls = {
    "d_year": 1,
    "s_region": 1,
    "c_region": 1,
    "p_mfgr": 1,

    "d_yearmonth": 2,
    "s_nation": 2,
    "c_nation": 2,
    "p_category": 2,

    "s_city": 3,
    "c_city": 3,
    "p_brand1": 3,
}

mu = {1: 0.05, 2: 0.10, 3: 0.20}

def weight(a: str) -> float:
    bits = ceil(log2(domain[a]))
    per_row_bytes = bits / 8.0 + mu[cls[a]]
    return factor * per_row_bytes

weights = {a: weight(a) for a in domain}

# ----------------------------
# 2. Standard SSB workload
#    R_i(q): referenced attrs
#    P_i(q): predicate attrs
#
# Updated per your request:
# - ignore conjunctive predicates
# - ignore BETWEEN predicates
# ----------------------------

W = {
    "Q1.1": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
    },
    "Q1.2": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
    },
    "Q1.3": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
    },
    "Q2.1": {
        "date": {"R": {"d_year"}, "P": set()},
        "supplier": {"R": {"s_region"}, "P": {"s_region"}},
        "part": {"R": {"p_category", "p_brand1"}, "P": {"p_category"}},
    },
    "Q2.2": {
        "date": {"R": {"d_year"}, "P": set()},
        "supplier": {"R": {"s_region"}, "P": {"s_region"}},
        "part": {"R": {"p_brand1"}, "P": set()},  # ignore BETWEEN on p_brand1
    },
    "Q2.3": {
        "date": {"R": {"d_year"}, "P": set()},
        "supplier": {"R": {"s_region"}, "P": {"s_region"}},
        "part": {"R": {"p_brand1"}, "P": {"p_brand1"}},
    },
    "Q3.1": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
        "customer": {"R": {"c_region", "c_nation"}, "P": {"c_region"}},
        "supplier": {"R": {"s_region", "s_nation"}, "P": {"s_region"}},
    },
    "Q3.2": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
        "customer": {"R": {"c_nation", "c_city"}, "P": {"c_nation"}},
        "supplier": {"R": {"s_nation", "s_city"}, "P": {"s_nation"}},
    },
    "Q3.3": {
        "date": {"R": {"d_year"}, "P": {"d_year"}},
        "customer": {"R": {"c_city"}, "P": {"c_city"}},
        "supplier": {"R": {"s_city"}, "P": {"s_city"}},
    },
    "Q3.4": {
        "date": {"R": {"d_year", "d_yearmonth"}, "P": {"d_yearmonth"}},
        "customer": {"R": {"c_city"}, "P": {"c_city"}},
        "supplier": {"R": {"s_city"}, "P": {"s_city"}},
    },
    "Q4.1": {
        "date": {"R": {"d_year"}, "P": set()},
        "customer": {"R": {"c_region", "c_nation"}, "P": {"c_region"}},
        "supplier": {"R": {"s_region"}, "P": {"s_region"}},
        "part": {"R": {"p_mfgr"}, "P": set()},  # ignore conjunctive predicate on p_mfgr
    },
    "Q4.2": {
        "date": {"R": {"d_year"}, "P": set()},  # ignore conjunctive predicate on d_year
        "customer": {"R": {"c_region"}, "P": {"c_region"}},
        "supplier": {"R": {"s_region", "s_nation"}, "P": {"s_region"}},
        "part": {"R": {"p_mfgr", "p_category"}, "P": set()},  # ignore conjunctive predicate on p_mfgr
    },
    "Q4.3": {
        "date": {"R": {"d_year"}, "P": set()},  # ignore conjunctive predicate on d_year
        "customer": {"R": {"c_region"}, "P": {"c_region"}},
        "supplier": {"R": {"s_nation", "s_city"}, "P": {"s_nation"}},
        "part": {"R": {"p_category", "p_brand1"}, "P": {"p_category"}},
    },
}

# ----------------------------
# 3. Utility
# ----------------------------

def delta_for_query(query_spec, G):
    score = 0
    for _, spec in query_spec.items():
        R = spec["R"]
        P = spec["P"]
        full = R.issubset(G)
        if full:
            score += 1
        else:
            score += len(P.intersection(G))
    return score

def U(G):
    return sum(delta_for_query(qspec, G) for qspec in W.values())

def total_weight(G):
    return sum(weights[a] for a in G)

# ----------------------------
# 4. Enumerate feasible sets
# ----------------------------

attrs = list(domain.keys())

best_u = -1
best_sets = []

for r in range(len(attrs) + 1):
    for combo in combinations(attrs, r):
        G = set(combo)
        w = total_weight(G)
        if w <= beta + 1e-12:
            u = U(G)
            if u > best_u:
                best_u = u
                best_sets = [(G, w, u)]
            elif u == best_u:
                best_sets.append((G, w, u))

best_sets = sorted(best_sets, key=lambda x: (x[1], sorted(list(x[0]))))

print("Best utility:", best_u)
print("Optimal feasible sets:")
for G, w, u in best_sets:
    print(f"  U={u}, w={w:.4f}, G={sorted(G)}")

G_emp = {"d_year", "s_region", "c_region", "s_nation"}
print("\nEmpirical set:")
print(f"  U(G_emp) = {U(G_emp)}")
print(f"  w(G_emp) = {total_weight(G_emp):.4f}")
print(f"  G_emp    = {sorted(G_emp)}")
