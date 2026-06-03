import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

def get_flip_labels(y, q, rate):
    n = len(y)
    n_flip = int(np.floor(n * rate))
    # We slice q[n:] to look at the weights assigned to the inverted labels
    idx_flip = np.argsort(q[n:])[::-1][:n_flip]
    y_flip = y.copy()
    y_flip[idx_flip] = -y_flip[idx_flip]
    return y_flip

def solveLP_analytical(eps, psi, C):
    """
    Replaces scipy.optimize.linprog with an exact $O(N \log N)$ analytical solution.
    """
    n2 = len(eps)
    n = n2 // 2

    # Objective coefficients: c = eps - psi
    c = eps - psi
    c1, c2 = c[:n], c[n:]

    # Since q1 + q2 = 1, substituting q1 = 1 - q2 into the objective gives:
    # minimize: sum(c1) + sum((c2 - c1) * q2)
    # So we only need to minimize v * q2, where v = c2 - c1
    v = c2 - c1
    
    q = np.zeros(n2)
    budget = n * C

    # Sort indices to greedily pick the most negative values of v
    sorted_indices = np.argsort(v)

    q2 = np.zeros(n)
    for idx in sorted_indices:
        # Stop if taking more items increases the objective, or if budget is empty
        if v[idx] >= 0 or budget <= 0:
            break
        take = min(1.0, budget)
        q2[idx] = take
        budget -= take

    # Enforce the equality constraint q1 + q2 = 1
    q[n:] = q2
    q[:n] = 1.0 - q2
    
    return q, "Analytical Solution Reached"

def solveQP_optimized(q, X_train, y_train, rate, svc_params):
    """
    Avoids duplicating X_train, calculating the decision function only once.
    """
    y_adv = get_flip_labels(y_train, q, rate)

    clf = SVC(**svc_params)
    clf.fit(X_train, y_adv)
    
    # Calculate margin ONLY for n samples, not 2n samples
    f_X = clf.decision_function(X_train)
    
    # Calculate hinge loss for original labels and inverted labels directly
    eps1 = np.maximum(0, 1 - f_X * y_train)
    eps2 = np.maximum(0, 1 + f_X * y_train) # Equivalent to 1 - f_X * (-y_train)
    
    eps = np.concatenate((eps1, eps2))
    return eps

def alfa(X_train, y_train, rate, svc_params, max_iter=5):
    clf = SVC(**svc_params)
    clf.fit(X_train, y_train)

    # Initial decision function
    f_X = clf.decision_function(X_train)

    # Calculate initial psi (hinge loss) without matrix duplication
    psi1 = np.maximum(0, 1 - f_X * y_train)
    psi2 = np.maximum(0, 1 + f_X * y_train)
    psi = np.concatenate((psi1, psi2))
    
    eps = np.zeros_like(psi)

    pbar = tqdm(range(max_iter), ncols=100)
    for _ in pbar:
        # 1. Solve the LP analytically instantly
        q, msg = solveLP_analytical(eps, psi, rate)
        pbar.set_postfix({'Optimizer': msg})
        
        # 2. Retrain SVM and get new margins with half the memory/compute
        eps = solveQP_optimized(q, X_train, y_train, rate, svc_params=svc_params)

    y_flip = get_flip_labels(y_train, q, rate)
    return y_flip