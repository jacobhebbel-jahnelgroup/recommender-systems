import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class MatrixFactorization:

    def __init__(self):

        self._errorMeasure = None
        self._learningAlgorithm = None

    def errorMeasure(self, measure):
        assert measure is not None, "Provide a function"
        self._errorMeasure = measure
        return self

    def learningAlgorithm(self, learning):
        assert learning is not None, "Provide a function"
        self._learningAlgorithm = learning
        return self

    def build(self):

        # check parameters are initialzied
        assert self._errorMeasure is not None, "Provide an error measure"
        assert self._learningAlgorithm is not None, "Provide an update rule"

        return Algorithm(self._errorMeasure, self._learningAlgorithm)

class Algorithm:

    def __init__(self, errorMeasure, learningAlgorithm):
        """
        Instantiates an algorithm with the specified architecture. Feature
        dimension is no longer fixed here since model selection needs to
        try multiple values of it - see init_params(feat_dim).
        """
        self._errorMeasure = errorMeasure
        self._learningAlgorithm = learningAlgorithm
        self._FEAT_DIM = None

        self.load_data()
        self._sync_ratings()

    def load_data(self):
        from surprise import Dataset
        from surprise.model_selection import train_test_split

        self._data = Dataset.load_builtin("ml-100k")
        trainval, self.testset = train_test_split(self._data, test_size=0.2, random_state=42)
        self.trainset, self.valset = self._split_validation(trainval, val_size=0.125, random_state=42)

        print(f'Trainset stats:')
        print(f'Number of users:    {self.trainset.n_users}')
        print(f'Number of items:    {self.trainset.n_items}')
        print(f'Number of ratings:  {self.trainset.n_ratings}')
        print(f'Number of val obs:  {len(self.valset)}')
        print(f'Global rating mean: {self.trainset.global_mean}')

    def _split_validation(self, trainset, val_size, random_state):
        """
        Carves a validation set out of `trainset`. `trainset` has no raw
        ratings file of its own to feed back through surprise's
        train_test_split, so its ratings are pulled out via build_testset(),
        shuffled, and reassembled into a smaller Trainset. `self._data` only
        lends its Reader/rating_scale here - it doesn't need to contain
        these ratings itself.

        Returns:
        new_trainset: a Trainset built from the non-validation ratings
        val_raw: a list of (raw_uid, raw_iid, rating) tuples, usable
                 directly as a testset via parse_dataset(..., is_trainset=False)
        """
        raw_ratings = trainset.build_testset()

        rng = np.random.RandomState(random_state)
        shuffled = rng.permutation(len(raw_ratings))
        n_val = int(len(raw_ratings) * val_size)
        val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]

        train_raw = [(*raw_ratings[i], None) for i in train_idx]
        val_raw = [raw_ratings[i] for i in val_idx]

        new_trainset = self._data.construct_trainset(train_raw)
        return new_trainset, val_raw

    def refit_on_trainval(self, feat_dim):
        """
        Note* Look for any data leakage
        Rebuilds self.trainset from train+val combined (used once model
        selection is done, so the final fit sees every non-test rating)
        and reinitializes params for the chosen feature dimension.
        """
        combined_raw = [(*r, None) for r in self.trainset.build_testset()]
        combined_raw += [(*r, None) for r in self.valset]
        self.trainset = self._data.construct_trainset(combined_raw)
        self._sync_ratings()
        self.init_params(feat_dim)

    def _sync_ratings(self):
        """
        Rebuilds the dense ratings matrix and summary stats from
        self.trainset. Independent of feature dimension, so it only needs
        to run when self.trainset itself changes (initial load, refit).
        """
        self.NUM_USERS = self.trainset.n_users
        self.NUM_ITEMS = self.trainset.n_items
        self.NUM_RATINGS = self.trainset.n_ratings
        self.mean = self.trainset.global_mean

        self.ratings_matrix = np.zeros((self.NUM_USERS, self.NUM_ITEMS))
        for inner_user, inner_item, rating in self.trainset.all_ratings():
            self.ratings_matrix[inner_user, inner_item] = rating

    def init_params(self, feat_dim):
        """
        (Re)initializes Q/P/Bq/Bp for the given feature dimension. Kept
        separate from _sync_ratings so model selection can retry different
        feature dimensions without re-deriving the ratings matrix each time.
        """
        self._FEAT_DIM = feat_dim

        # init matrices on a norm. dist with mean 0 and stddev based on num features
        # init biases at 0 to keep them deterministic-esque
        std_dev = 1.0 / np.sqrt(feat_dim)

        self.Q = np.random.normal(loc=0.0, scale=std_dev, size=(self.NUM_USERS, feat_dim))
        self.P = np.random.normal(loc=0.0, scale=std_dev, size=(self.NUM_ITEMS, feat_dim))
        self.Bq = np.zeros(self.NUM_USERS)
        self.Bp = np.zeros(self.NUM_ITEMS)

    def fit(self, **kwargs):
        """
        Calls the learning algorithm to create features for each user and item
        Args depend on the algorithm, check with the specified algorithm's documentation
        Does not return anything
        """
        self._learningAlgorithm(**kwargs)
    
    def eval(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        """
        Args:
        users: a list of indices to the users matrix
        items: a lit of indices to the items matrix
        ratings: a sparse truth matrix where user-item indices are nonzero

        Returns:
        rmse: root mean squared error between preds and truths
        preds: the predicted ratings matrix
        """ 
        
        u = self.mean
        Q, P = self.Q[users], self.P[items]
        Bq, Bp = self.Bq[users], self.Bp[items]

        interactions = Q @ P.T
        biases = Bq[:, np.newaxis] + Bp[np.newaxis, :]
        R_hat = u + biases + interactions
        pred_ratings = np.clip(R_hat, 1.0, 5.0)
        
        nonzero_mask = ratings > 0
        err = self._errorMeasure(ratings[nonzero_mask], pred_ratings[nonzero_mask])
        return pred_ratings, err


def rmse(expected: np.ndarray, predicted: np.ndarray):    
    """
    Uses a vectorized operation to compute the 
    square difference between expected and predicted observations
    """
    return np.sqrt(np.mean((expected - predicted) ** 2))


def mse(expected: np.ndarray, predicted: np.ndarray):
    """
    Computes the mean square error
    between expected and predicted tensors
    """
    return np.mean((expected - predicted) ** 2)


def als(R: np.ndarray, P: np.ndarray, Q: np.ndarray, Bp: np.ndarray, Bq: np.ndarray, U: float, C: float, max_itr: int = 100, tol: float = 1e-4):
    """
    Objective Function: minimize Q, P, Bq, Bp with a regularization term
    Strategy: Fix Q or P and solve analytically
    """

    def fixed_quadratic_solver( 
        M1: np.ndarray, M2: np.ndarray, 
        B1: np.ndarray, B2: np.ndarray, 
        axis: int
    ):
        """
        Iteratively solves the quadratic program for M1, B1
        M1: The optimizing embeddings
        M2: The fixed embeddings
        B1: The optimizing biases
        B2: The fixed biases
        axis: Which type of problem to solve 
        """
        num_subproblems, K = M1.shape
        I_k1 = np.eye(K + 1)
        M2_aug = np.hstack([np.ones((M2.shape[0], 1)), M2])
        for i in range(num_subproblems):
            # iterate over the item index

            # determine which variables are related to this subproblem
            if axis == 0:
                dependents = np.where(R[i, :] > 0)[0]
                R_raw = R[i, dependents]
            else:
                dependents = np.where(R[:, i] > 0)[0]
                R_raw = R[dependents, i]

            if len(dependents) == 0: continue
            
            # choose fixed variables related to the subproblem
            M2i = M2_aug[dependents, :]
            B2i = B2[dependents]
            Ri = R_raw - U - B2i

            # Scale regularization by number of interactions for this subproblem
            # Reg[0] refers to the bias-specific regularization: regularize lightly compared to features
            # Comparing a fixed bias regularization vs one that scales with C and interaction size
            FIXED_BIAS_C = 0.01
            Reg = I_k1 * C * len(dependents)
            Reg[0, 0] = FIXED_BIAS_C # *= 0.01
            
            A = M2i.T @ M2i + Reg
            b = M2i.T @ Ri
            optimal = np.linalg.solve(A, b)
            
            # remember optimal is packaged bias first then embedding
            B1[i] = optimal[0]
            M1[i, :] = optimal[1:]

    def train_rmse():
        mask = R > 0
        R_hat = U + Bq[:, np.newaxis] + (Bp + Q @ P.T)
        return np.sqrt(np.mean((R[mask] - R_hat[mask]) ** 2))

    prev_rmse = train_rmse()
    for _ in range(max_itr):
        
        # optimize Q, Bq
        fixed_quadratic_solver(
            M1=Q, M2=P, B1=Bq, B2=Bp, axis=0
        )

        # optimize P, Bp
        fixed_quadratic_solver(
            M1=P, M2=Q, B1=Bp, B2=Bq, axis=1
        )

        # stop if loss is below threshold
        curr_rmse = train_rmse()
        if abs(prev_rmse - curr_rmse) < tol:
            break
        prev_rmse = curr_rmse


def _fit_and_score(c, feat_dim, seed, R, mean, val_users, val_items, val_ratings, learning_algorithm, error_measure):
    """
    Runs in a worker process: builds a fresh Q/P/Bq/Bp sized for feat_dim
    ((c, feat_dim) pairs share no state), fits with the given C, and scores
    against the validation set. Module-level (not nested in a caller) so it
    can be pickled and sent to worker processes - ProcessPoolExecutor can
    only pickle top-level functions, not closures.
    """
    rng = np.random.RandomState(seed)
    num_users, num_items = R.shape
    std_dev = 1.0 / np.sqrt(feat_dim)

    Q = rng.normal(loc=0.0, scale=std_dev, size=(num_users, feat_dim))
    P = rng.normal(loc=0.0, scale=std_dev, size=(num_items, feat_dim))
    Bq = np.zeros(num_users)
    Bp = np.zeros(num_items)

    learning_algorithm(R=R, P=P, Q=Q, Bp=Bp, Bq=Bq, U=mean, C=c)

    Qv, Pv = Q[val_users], P[val_items]
    Bqv, Bpv = Bq[val_users], Bp[val_items]
    R_hat = mean + Bqv[:, np.newaxis] + Bpv[np.newaxis, :] + Qv @ Pv.T
    pred_ratings = np.clip(R_hat, 1.0, 5.0)

    nonzero_mask = val_ratings > 0
    err = error_measure(val_ratings[nonzero_mask], pred_ratings[nonzero_mask])
    return c, feat_dim, err


# NOTE: THIS WAS WRITTEN BY AI
def choose_hyperparams_parallel(
    algo, val_users, val_items, val_ratings,
    c_candidates, feat_dim_candidates,
    max_workers=None):
    """
    Parallel grid search over every (C, feat_dim) pair: since regularization
    strength and feature dimension interact (more latent dimensions need
    more regularization to avoid overfitting), they can't be tuned
    independently - each pair fits a fresh Q/P/Bq/Bp, so every pair runs in
    its own process.

    Must be called under `if __name__ == "__main__":` — on Windows,
    ProcessPoolExecutor uses the "spawn" start method, which re-imports this
    module in every worker; without the guard, each worker would re-run the
    whole script's top-level code instead of just the worker function.
    """
    R = algo.ratings_matrix
    mean = algo.mean
    learning_algorithm = algo._learningAlgorithm
    error_measure = algo._errorMeasure

    grid = [(c, k) for k in feat_dim_candidates for c in c_candidates]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _fit_and_score, c, k, seed, R, mean,
                val_users, val_items, val_ratings,
                learning_algorithm, error_measure,
            ): (c, k)
            for seed, (c, k) in enumerate(grid)
        }
        for future in as_completed(futures):
            c, k, err = future.result()
            results[(c, k)] = err

    ordered_results = [(c, k, results[(c, k)]) for c, k in grid]
    for c, k, err in ordered_results:
        print(f'C={c:<6} K={k:<5} val RMSE={err:.4f}')

    best_C, best_K, best_rmse = min(ordered_results, key=lambda triple: triple[2])
    return best_C, best_K, best_rmse, ordered_results


# NOTE: THIS WAS WRITTEN BY AI
def assess_topk_stability(algo, val_users, val_items, val_ratings, ordered_results, top_k=8, n_seeds=8, max_workers=None):
    """
    Re-fits only the top_k best-performing (C, K) pairs from a sweep,
    n_seeds times each with a fresh random init, to check whether the
    ranking among top performers reflects a real effect or just
    initialization/validation-split noise. Restricted to the top
    performers (rather than every grid point) since repeating the full
    grid n_seeds times would multiply an already large sweep's runtime.
    """
    R = algo.ratings_matrix
    mean = algo.mean
    learning_algorithm = algo._learningAlgorithm
    error_measure = algo._errorMeasure

    top_pairs = [(c, k) for c, k, _ in sorted(ordered_results, key=lambda t: t[2])[:top_k]]

    results = {pair: [] for pair in top_pairs}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for pair_idx, (c, k) in enumerate(top_pairs):
            for trial in range(n_seeds):
                # offset well clear of the sweep's own seeds so trials don't
                # accidentally reuse an initialization from the grid search
                seed = 1_000_000 + pair_idx * n_seeds + trial
                future = pool.submit(
                    _fit_and_score, c, k, seed, R, mean,
                    val_users, val_items, val_ratings,
                    learning_algorithm, error_measure,
                )
                futures[future] = (c, k)
        for future in as_completed(futures):
            c, k, err = future.result()
            results[(c, k)].append(err)

    print(f'\nStability check on top {top_k} (C, K) pairs across {n_seeds} seeds each:')
    for (c, k), errs in sorted(results.items(), key=lambda kv: np.mean(kv[1])):
        errs = np.array(errs)
        print(f'C={c:<6} K={k:<5} mean RMSE={errs.mean():.4f}  std={errs.std():.4f}  min={errs.min():.4f}  max={errs.max():.4f}')

    return results


# NOTE: THIS WAS WRITTEN BY AI
def plot_topk_stability(stability_results, out_path="topk_stability.png"):
    """
    Boxplot of validation RMSE across seeds for each top-performing (C, K)
    pair, sorted by mean RMSE - lets you see whether the "winning" config's
    spread actually separates from the runners-up or just overlaps with them.
    """
    import matplotlib.pyplot as plt

    items = sorted(stability_results.items(), key=lambda kv: np.mean(kv[1]))
    labels = [f'C={c:.3g}\nK={k}' for (c, k), _ in items]
    data = [errs for _, errs in items]

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 1.1), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Stability of top performers across random seeds')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved stability plot to {out_path}')


# NOTE: THIS WAS WRITTEN BY AI
def plot_sweep_results(sweep_results, best_C, best_K, out_path="model_selection.png"):
    """
    Plots validation RMSE vs C (log scale), one line per feature dimension K,
    with the (best_C, best_K) point highlighted.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    by_k = {}
    err_by_pair = {}
    for c, k, err in sweep_results:
        by_k.setdefault(k, []).append((c, err))
        err_by_pair[(c, k)] = err

    # K is an ordered numeric quantity, and there are too many values for a
    # discrete per-line legend - encode it as a colormap + colorbar instead.
    ks = sorted(by_k)
    norm = mcolors.Normalize(vmin=min(ks), vmax=max(ks))
    cmap = plt.colormaps['viridis']

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ks:
        points = sorted(by_k[k])
        cs = [p[0] for p in points]
        errs = [p[1] for p in points]
        ax.plot(cs, errs, marker='o', markersize=3, color=cmap(norm(k)))

    ax.scatter([best_C], [err_by_pair[(best_C, best_K)]],
                s=90, facecolors='none', edgecolors='black', linewidths=1.5, zorder=5,
                label=f'Best (C={best_C:.3g}, K={best_K})')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('K (feature dimension)')

    ax.set_xscale('log')
    ax.set_xlabel('C (regularization strength, log scale)')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Model selection: validation RMSE vs C and K')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved model selection plot to {out_path}')


# NOTE: THIS WAS WRITTEN BY AI
def grid_search(algo: Algorithm, 
    val_users: np.ndarray, val_items: np.ndarray, val_ratings: np.ndarray,
    use_parallel: bool = True, use_handcrafted: bool = True, 
    plot_sweep: bool = True, assess_topk: bool = True, plot_topk: bool = True,
    custom_c: np.ndarray = None, custom_k: np.ndarray = None):
    """
    Wrapper function for grid search functionality
    
    Args:
    - use_parallel:     run fit-eval jobs for parameter combinations in parallel
    - use_handcrafted:  use the on-hand handcrafted features instead of supplying custom ones
    - plot_sweep:       If a plot of parameter performance should be made
    - assess_topk:      Runs a random-seed test to verify topK results
    - plot_topk:        plots a box-and-whisker plot to explain topk stability
    - custom_c:         user-provided list for the regularization parameter
    - custom_k:         user-provided list for the feature dimension parameter
    - algo:             the Algorithm to do grid search on
    - val_users         np user indices into the ratings matrix
    - val_items         np item indices into the ratings matrix
    - val_ratings       complete sparse np matrix of user-item ratings 
    """

    # handcrafted features
    handcrafted_c = np.concatenate([
        np.arange(0.01, 0.1, 0.01),
        np.arange(0.1, 1, 0.1)
    ])
    handcrafted_k = np.concatenate([
        np.arange(1, 10, 1),
        np.arange(10, 101, 10)
    ])

    # parameter selection logic
    candidate_c = custom_c
    candidate_k = custom_k
    if use_handcrafted:
        candidate_c = handcrafted_c
        candidate_k = handcrafted_k

    search_results = None
    if use_parallel:
        search_results = choose_hyperparams_parallel(
            algo, val_users, val_items, val_ratings,
            candidate_c, candidate_k
        )
    else:
        raise RuntimeException()
    
    c_star, k_star, rmse_star, results = search_results

    if plot_sweep:
        plot_sweep_results(results, c_star, k_star)
    
    if assess_topk:
        stability_results = assess_topk_stability(algo, val_users, val_items, val_ratings, results)

        if plot_topk:
            plot_topk_stability(stability_results)
    
    return c_star, k_star, rmse_star, results