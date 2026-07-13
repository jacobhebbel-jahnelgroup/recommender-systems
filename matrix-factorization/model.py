import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

from lib import MatrixFactorization, Algorithm
from lib import grid_search, rmse, als

def parse_dataset(trainset, testset = None, is_trainset: bool = True):
    """
    Returns user indices, item indices, and a complete sparse ratings matrix
    """

    def parse_trainset(dataset):
        
        num_users, num_items = dataset.n_users, dataset.n_items
        ratings_matrix = np.zeros((num_users, num_items))
        for inner_user, inner_item, rating in dataset.all_ratings():
            ratings_matrix[inner_user, inner_item] = rating

        return np.arange(num_users), np.arange(num_items), ratings_matrix
    
    def parse_testset(trainset, testset):
                
        num_users, num_items = trainset.n_users, trainset.n_items
        ratings_matrix = np.zeros((num_users, num_items))
        for uid, iid, rating in testset:

            try:
                iuid = trainset.to_inner_uid(uid)
                iiid = trainset.to_inner_iid(iid)
            except ValueError:
                # user or item unseen in the trainset (cold start); skip
                continue

            ratings_matrix[iuid, iiid] = rating

        return np.arange(num_users), np.arange(num_items), ratings_matrix

    # choose the right parsing method
    if is_trainset: return parse_trainset(trainset)
    else:           return parse_testset(trainset, testset)

def train_on_ratings(algo, c):
    """
    Simple wrapper to avoid writing the full algo.fit call
    """
    algo.fit(
        R=algo.ratings_matrix,
        Q=algo.Q,
        P=algo.P,
        Bq=algo.Bq,
        Bp=algo.Bp,
        U=algo.mean,
        C=c
    )


if __name__ == "__main__":

    # construct our matrix algorithm
    mf = MatrixFactorization()
    algorithm = mf.errorMeasure(rmse).learningAlgorithm(als).build()


    # GETTING DATA - trainset is passed to minimize cold entries
    val_users, val_items, val_ratings = parse_dataset(algorithm.trainset, algorithm.valset, is_trainset=False)
    

    # MODEL SELECTION: sweep (C, feature dimension) jointly, scoring each
    # candidate pair against the validation set
    best_C, best_K, best_val_rmse, sweep_results = grid_search(
        use_parallel=True,
        use_handcrafted=True,
        plot_sweep=True,
        assess_topk=True,
        plot_topk=True,
        algo=algorithm,
        val_users=val_users,
        val_items=val_items,
        val_ratings=val_ratings
    )


    # FINAL FIT: retrain on train+val combined with the winning (C, K), so the
    # model sees every rating except the held-out test set
    algorithm.refit_on_trainval(best_K)
    train_on_ratings(algorithm, best_C)

    # Re-derive train/test indices against algorithm.trainset now, after the
    # refit above reassigned it (train+val combined) and remapped raw ids to
    # inner ids - reusing indices parsed against the pre-refit trainset would
    # silently pair ratings with the wrong rows of Q/P.
    train_users, train_items, train_ratings = parse_dataset(algorithm.trainset, is_trainset=True)
    test_users, test_items, test_ratings = parse_dataset(algorithm.trainset, algorithm.testset, is_trainset=False)

    # TRAIN PASS: Compute the in-sample error on the dataset
    # none of this data is held out, so it should score high
    _, train_rmse = algorithm.eval(train_users, train_items, train_ratings)

    # TEST PASS: This gets the model's performance on the unseed test data
    # This result is the out of sample error, a good guess at the model's accuracy
    test_preds, test_rmse = algorithm.eval(test_users, test_items, test_ratings)

    print(f'\nTrain RMSE:     | {train_rmse:.4f}')
    print(f'Test RMSE:      | {test_rmse:.4f}')
    print(f'Train/Test gap: | {abs(train_rmse - test_rmse):.4f}')

    # SANITY TESTS:
    # make a prediction matrix of just the mean
    test_mask = test_ratings > 0
    baseline_preds = np.full_like(test_ratings, algorithm.mean, dtype=float)
    baseline_rmse = rmse(test_ratings[test_mask], baseline_preds[test_mask])
    
    # compare baseline with test: test >> baseline ==> model has learned something intelligent
    print(f'\nBaseline (predict global mean) RMSE: {baseline_rmse:.4f}')
    print(f'Improvement over baseline: {baseline_rmse - test_rmse:.4f}')

    # Precision performance
    abs_err = np.abs(test_ratings[test_mask] - test_preds[test_mask])
    print(f'% test predictions within 0.5 stars: {(abs_err <= 0.5).mean() * 100:.1f}%')
    print(f'% test predictions within 1.0 stars: {(abs_err <= 1.0).mean() * 100:.1f}%')

    n_test_raw = len(algorithm.testset)
    n_test_scored = int(test_mask.sum())
    if n_test_scored < n_test_raw:
        print(f'Skipped {n_test_raw - n_test_scored}/{n_test_raw} test ratings (cold-start users/items)')