import numpy as np

def dirichlet_indices(labels, num_partitions: int, alpha: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    client_indices = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)

        p = rng.dirichlet(alpha * np.ones(num_partitions))
        counts = (p * len(cls_idx)).astype(int)

        diff = len(cls_idx) - counts.sum()
        if diff > 0:
            rem_order = np.argsort(-p)[:diff]
            counts[rem_order] += 1

        start = 0
        for i, k in enumerate(counts):
            if k > 0:
                client_indices[i].extend(cls_idx[start:start + k].tolist())
                start += k

    for i in range(num_partitions):
        rng.shuffle(client_indices[i])

    return client_indices
