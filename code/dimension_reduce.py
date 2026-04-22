import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    input_path: str | Path,
    output_path: str | Path = None,
    variance: float = 0.95,
    scale: bool = True,
):
    """
    Apply PCA to embedding vectors stored in a .npz file.

    Parameters
    ----------
    input_path : path to .npz file containing:
        - embeddings (N, D)
        - row_ids, tweet_ids, timestamps

    output_path : where to save result (auto-generated if None)

    variance : target explained variance (default = 0.95)

    scale : whether to standardize features before PCA
        - True  → recommended for local embeddings
        - False → for OpenAI embeddings

    Returns
    -------
    embeddings_pca : (N, K)
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"Loading: {input_path}")
    data = np.load(input_path, allow_pickle=False)

    X          = data["embeddings"].astype(np.float64)
    row_ids    = data["row_ids"]
    tweet_ids  = data["tweet_ids"]
    timestamps = data["timestamps"]

    N, D = X.shape
    print(f"Embeddings: {N} x {D}")

    # optional scaling 
    if scale:
        print("Scaling embeddings...")
        X = StandardScaler().fit_transform(X)

    # PCA 
    print(f"Running PCA (target variance={variance})...")
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X).astype(np.float32)

    K = pca.n_components_
    explained = float(pca.explained_variance_ratio_.sum())

    print(f"Reduced: {D} → {K} dims ({explained:.2%} variance)")

    # auto-generate output path 
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_pca.npz"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings_pca=X_pca,
        row_ids=row_ids,
        tweet_ids=tweet_ids,
        timestamps=timestamps,
        n_components=np.array(K, dtype=np.int32),
        explained_var=np.array(explained, dtype=np.float32),
    )

    print(f"Saved → {output_path}")

    return X_pca


# simple loader 
def load_pca(path: str | Path):
    data = np.load(path, allow_pickle=False)
    return (
        data["embeddings_pca"],
        data["row_ids"],
        data["tweet_ids"],
        data["timestamps"],
    )


if __name__ == "__main__":
    # Example usage
    run_pca("data/vector_embeddings/all-MiniLM-L6-v2/all-MiniLM-L6-v2.npz")