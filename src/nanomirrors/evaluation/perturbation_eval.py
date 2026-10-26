from typing import Tuple, List, Dict, Literal

import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import anndata as ad
from scipy.spatial.distance import cdist, cosine, pdist
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def _group_results_by_condition(
        preds: np.ndarray, 
        gts: np.ndarray,
        perturbations: List[str], 
        covariates: List[dict],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Group the predictions and ground truth values by condition.
    The condition is defined by the combination of perturbations and covariates.
    Each condition is `{`perturbation`:pert_value}_{`covariate1`:cov_value1}_{`covariate2`:cov_value2}_...` where the covariates are concatenated in order.

    Args:
        preds: Predicted values, shape (n_cells, n_genes).
        gts: Ground truth values, shape (n_cells, n_genes).
        perturbations: List of perturbation labels corresponding to each sample.
        covariates: List of dictionaries containing covariate information for each sample, excluding the `perturbation`.
        e.g. [{"batch": 1, "cell_line": "A"}, {"batch": 2, "cell_line": "B"}, ...].

    Returns:
        A tuple of two dictionaries:
        - The first dictionary maps each condition to the corresponding predicted values.
        - The second dictionary maps each condition to the corresponding ground truth values.
    """
    preds_groups = {}
    gt_groups = {}

    
    for i in range(len(preds)):
        pred = preds[i]
        gt = gts[i]
        perturbation = perturbations[i]
        if covariates:
            covariate = covariates[i]
        else:
            covariate = None

        condition = f"perturbation:{perturbation}"
        if covariate:
            condition += "_" + "_".join(f"{k}:{v}" for k, v in covariate.items())
        if condition not in preds_groups:
            preds_groups[condition] = []
            gt_groups[condition] = []
        preds_groups[condition].append(pred)
        gt_groups[condition].append(gt)

    for condition in preds_groups:
        preds_groups[condition] = np.array(preds_groups[condition])
        gt_groups[condition] = np.array(gt_groups[condition])

    return preds_groups, gt_groups

def get_de_results_by_condition(
        adata: ad.AnnData, 
        condition: str, 
        fdr: float = 0.05,
        de_column: str = "de",
        de_result_key: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the differentially expressed genes for a given condition.

    Args:
        adata: AnnData object containing differential expression results.
        condition: Condition string to filter the DE results.
        fdr: False discovery rate threshold.
        de_column: Column name in `adata.var` containing DE p-values or statistics.
        de_result_key: Key in `adata.uns` where DE results are stored.

    Returns:
        A tuple containing two lists:
        - The first list contains all DEGs.
        - The second list contains DEGs passing the FDR threshold.
    """
    cond_keys, cond_values = zip(*(item.split(":") for item in condition.split("_")))
    cond_dict = dict(zip(cond_keys, cond_values))
    all_de_results = adata.uns[de_column][de_result_key]

    mask = None
    for key, value in cond_dict.items():
        if mask is None:
            mask = (all_de_results[key] == value)
        else:
            mask &= (all_de_results[key] == value)

    de_result = all_de_results[mask]
    de_result_significant = de_result[de_result["pvals_adj"] < fdr]

    return de_result, de_result_significant


def mse(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Compute the mean squared error between predictions and ground truth.

    Args:
        preds: Predicted values, shape (n_samples, n_features).
        gts: Ground truth values, shape (n_samples, n_features).

    Returns:
        Mean squared error.
    """
    assert preds.shape == gts.shape, "Predictions and ground truth must have the same shape."
    return np.mean((preds - gts) ** 2, axis=1).mean()

def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the energy distance between two sample distributions X and Y.

    Parameters
    ----------
    X : np.ndarray (N_X, D)
        First sample distribution.
    Y : np.ndarray (N_Y, D)
        Second sample distribution.

    Returns
    -------
    float
        Bias-corrected multivariate E-distance between X and Y.

    """
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return np.nan
    
    xy = cdist(X, Y, metric="euclidean").mean()
    xx = pdist(X, metric="euclidean").mean() # N(N-1) bias corrected
    yy = pdist(Y, metric="euclidean").mean()

    return 2 * xy - xx - yy


def compute_expression_fidelity_metrics_by_group(
    pred_groups: Dict[str, np.ndarray],
    gt_groups: Dict[str, np.ndarray],
    ctrl_adata: ad.AnnData,
    fdr=0.05,
    top_deg=100,
) -> Dict[str, Dict[str, float]]:
    """
    Compute expression fidelity metrics for the given predicted and ground truth groups.

    Args:
        pred_groups: Dictionary mapping each condition to the corresponding predicted values.
        gt_groups: Dictionary mapping each condition to the corresponding ground truth values.
        perturbations: List of perturbation labels corresponding to each sample.
        covariates: List of dictionaries containing covariate information for each sample.
        top_deg: Number of top DE genes to consider for top DE gene metrics.
        ctrl_adata: Control AnnData object.
        fdr: False discovery rate threshold.

    Returns:
        A dictionary mapping condition to the following expression fidelity metrics.
        - `mse_all_genes`
        - `pearson_delta_all_genes`
        - `mse_de_genes`
        - `pearson_delta_de_genes`
        - `energy_distance`
    """
    metrics_by_group = {}

    for condition in tqdm(pred_groups, desc="Computing expression fidelity metrics by group"):
        preds = pred_groups[condition]
        gts = gt_groups[condition]

        ctrl_mean = np.asarray(ctrl_adata.X.mean(axis=0)).flatten()
        preds_delta = preds.mean(axis=0) - ctrl_mean
        gts_delta = gts.mean(axis=0) - ctrl_mean

        # 1. Compute metrics for all genes
        mse_all_genes = mse(preds, gts)
        pearson_delta_all_genes = pearsonr(preds_delta, gts_delta)[0]

        # 2. Compute metrics for DE genes
        de_result, de_result_significant = get_de_results_by_condition(ctrl_adata, condition, fdr=fdr)
        de_genes_sig = de_result_significant["gene"]
        print(f"Number of significant DE genes: {len(de_genes_sig)}")
        de_genes_sig_mask = ctrl_adata.var_names.isin(de_genes_sig)
        cond_top_deg = min(top_deg, len(de_genes_sig))
        top_de_genes = de_result_significant.sort_values("pvals_adj").head(cond_top_deg)["gene"]
        top_de_genes_mask = ctrl_adata.var_names.isin(top_de_genes)

        if len(de_genes_sig) == 0:
            print(f"[Warning] No significant DE genes found for condition {condition}. Skipping DE gene metrics.")
            continue

        preds_de = preds[:, de_genes_sig_mask]
        gts_de = gts[:, de_genes_sig_mask]
        print(f"preds_de shape: {preds_de.shape}, gts_de shape: {gts_de.shape}")
        
        mse_de_genes = mse(preds_de, gts_de)
        pearson_delta_de_genes = pearsonr(
            preds_delta[de_genes_sig_mask], 
            gts_delta[de_genes_sig_mask]
        )[0]

        preds_top_de = preds[:, top_de_genes_mask]
        gts_top_de = gts[:, top_de_genes_mask]
        print(f"preds_top_de shape: {preds_top_de.shape}, gts_top_de shape: {gts_top_de.shape}")
        
        mse_top_de_genes = mse(preds_top_de, gts_top_de)
        pearson_delta_top_de_genes = pearsonr(
            preds_delta[top_de_genes_mask], 
            gts_delta[top_de_genes_mask]
        )[0]

        # Compute energy distance
        e_dist = energy_distance(preds, gts)

        metrics_by_group[condition] = {
            "mse_all_genes": float(mse_all_genes),
            "pearson_delta_all_genes": float(pearson_delta_all_genes),
            "mse_de_genes": float(mse_de_genes),
            "pearson_delta_de_genes": float(pearson_delta_de_genes),
            "energy_distance": float(e_dist),
            "mse_top_de_genes": float(mse_top_de_genes),
            "pearson_delta_top_de_genes": float(pearson_delta_top_de_genes),
        }

    return metrics_by_group

def compute_differential_expression(
    exprs: np.ndarray,
    ctrl_adata: ad.AnnData,
    perturbation_col="perturbation",
    control_label="control",
    fdr=0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute differential expression results for the given expression values with respect to the control AnnData.

    Args:
        exprs (np.ndarray): input expression values, need to be in the same order as the genes in `ctrl_adata.var`.
        ctrl_adata (ad.AnnData): Control AnnData object.
        perturbation_col (str, optional): Column name for perturbations in the AnnData object. Defaults to "perturbation".
        control_label (str, optional): Label for the control condition. Defaults to "control".
        fdr (float, optional): False discovery rate threshold for identifying significant DE genes. Defaults to 0.05.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains all DE results.
            - The second DataFrame contains significant DE results based on the specified FDR threshold.
    """
    assert exprs.shape[1] == ctrl_adata.shape[1], "The number of genes in `exprs` must match the number of genes in `ctrl_adata`."
    assert ctrl_adata.obs[perturbation_col].isin([control_label]).all(), "Control AnnData should only contain control cells."

    if exprs.shape[0] <=1 :
        print(f"[warning] Not enough cells (num = {exprs.shape[0]}) to perform differential expression analysis. Skipping DE computation.")
        return pd.DataFrame(), pd.DataFrame()

    # concat the input expression values with the control AnnData
    exprs_adata = ad.AnnData(exprs)
    exprs_adata.var = ctrl_adata.var.copy()
    combined_adata = ad.concat([ctrl_adata, exprs_adata], join="outer")

    combined_adata.obs["_de_group"] = np.where(combined_adata.obs[perturbation_col] == control_label, "control", "pert")

    # perform DE analysis
    sc.tl.rank_genes_groups(
        combined_adata,
        groupby="_de_group",
        groups=["pert"],
        reference="control",
        method="wilcoxon",
        n_genes=combined_adata.n_vars,
        use_raw=False,
    )

    de = sc.get.rank_genes_groups_df(combined_adata, group="pert").rename(
            columns={"names": "gene", "scores": "score"})

    all_de_results = de
    all_de_results_significant = all_de_results[all_de_results["pvals_adj"] < fdr]

    return all_de_results, all_de_results_significant

def deg_set_overlap(pred_de_result_df: pd.DataFrame, gt_de_result_df: pd.DataFrame) -> float:
    """
    Compute the overlap between the predicted and ground truth sets of differentially expressed genes.
    Essentially, DEG Recall@N, where N is the number of ground truth DE genes.

    Args:
        pred_de_result_df (pd.DataFrame): Predicted DE results DataFrame.
        gt_de_result_df (pd.DataFrame): Ground truth DE results DataFrame.

    Returns:
        float: The fraction of overlapping DE genes between the predicted and ground truth sets.
    """
    
    if pred_de_result_df.empty or gt_de_result_df.empty:
        return 0.0
    
    pred_genes = set(pred_de_result_df["gene"])
    gt_genes = set(gt_de_result_df["gene"])
    if not gt_genes:
        return 0.0

    # sort by logfc, take the top len(gt_genes) predicted genes
    if len(pred_genes) > len(gt_genes):
        pred_de_result_df = pred_de_result_df.sort_values(by="logfoldchanges", ascending=False).head(len(gt_genes))
        pred_genes = set(pred_de_result_df["gene"])

    overlap_fraction = float(len(pred_genes & gt_genes)) / len(gt_genes)
    return overlap_fraction

def logfc_metrics(
    pred_de_result_df: pd.DataFrame,
    gt_de_result_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute log fold-change metrics between predicted and ground truth DE results.
    Only genes in the ground truth set are considered.

    Args:
        pred_de_result_df (pd.DataFrame): Predicted DE results DataFrame.
        gt_de_result_df (pd.DataFrame): Ground truth DE results DataFrame.

    Returns:
        dict with:
            - logfc_spearman: Spearman correlation of log fold changes.
            - direction_agreement: Fraction of genes with the same direction of log fold change.
    """
    genes2logfc_gt = dict(
        zip(gt_de_result_df["gene"], gt_de_result_df["logfoldchanges"])
    )

    pred_logfc = []
    gt_logfc = []

    for _, row in pred_de_result_df.iterrows():
        gene = row["gene"]

        if gene in genes2logfc_gt:
            pred_logfc.append(row["logfoldchanges"])
            gt_logfc.append(genes2logfc_gt[gene])

    if len(pred_logfc) == 0:
        return {
            "logfc_spearman": 0.0,
            "direction_agreement": 0.0,
        }

    pred_logfc = np.asarray(pred_logfc)
    gt_logfc = np.asarray(gt_logfc)

    direction = float(
        np.mean(np.sign(pred_logfc) == np.sign(gt_logfc))
    )

    if len(pred_logfc) < 2:
        spearman = np.nan
    else:
        spearman = float(
            spearmanr(pred_logfc, gt_logfc)[0]
        )

    return {
        "logfc_spearman": spearman,
        "direction_agreement": direction,
    }


def logfc_spearman(pred_de_result_df: pd.DataFrame, gt_de_result_df: pd.DataFrame) -> float:
    """
    Compute the Spearman correlation of log fold changes between the predicted and ground truth DE results. 
    Only genes in the ground truth set are considered.

    Args:
        pred_de_result_df (pd.DataFrame): Predicted DE results DataFrame.
        gt_de_result_df (pd.DataFrame): Ground truth DE results DataFrame.

    Returns:
        float: Spearman correlation of log fold changes.
    """
    genes2logfc_gt = dict(zip(gt_de_result_df["gene"], gt_de_result_df["logfoldchanges"]))

    pred_logfc = []
    gt_logfc = []
    for gene in pred_de_result_df["gene"]:
        if gene in genes2logfc_gt:
            pred_logfc.append(pred_de_result_df.loc[pred_de_result_df["gene"] == gene, "logfoldchanges"].values[0])
            gt_logfc.append(genes2logfc_gt[gene])

    if not pred_logfc:
        return 0.0
    return spearmanr(pred_logfc, gt_logfc)[0]


def compute_differential_expression_metrics_by_group(
    pred_groups: Dict[str, np.ndarray],
    gt_groups: Dict[str, np.ndarray],
    ctrl_adata: ad.AnnData,
    fdr=0.05,
):
    """
    Compute differential expression metrics for each condition.

    Args:
        pred_groups (Dict[str, np.ndarray]): A dictionary mapping condition to predicted expression values.
        gt_groups (Dict[str, np.ndarray]): A dictionary mapping condition to ground truth expression values.
        ctrl_adata (ad.AnnData): Control AnnData object.
        fdr (float, optional): False discovery rate threshold for identifying significant DE genes. Defaults to 0.05.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping condition to the following differential expression metrics:
            - `deg_overlap` 
            - `logfc_spearman`
            - `direction_agreement`
    """
    de_metrics_by_group = {}

    for condition in tqdm(pred_groups, desc="Computing differential expression metrics by group"):
        preds = pred_groups[condition]
        gts = gt_groups[condition]

        if preds.min() < 0:
            print(f"Warning: Negative values found in predictions for condition '{condition}'. Minimum value: {preds.min()}. Clipping to zero.")
            preds = np.clip(preds, a_min=0, a_max=None)

        # 0. Compute differential expression results for the ground truth and predicted data.
        gt_de_result, gt_de_result_sig= get_de_results_by_condition(ctrl_adata, condition, fdr=fdr)
        pred_de_result, pred_de_result_sig = compute_differential_expression(preds, ctrl_adata, fdr=fdr)

        # 1. Compute the overlap of differentially expressed genes.
        deg_overlap = deg_set_overlap(pred_de_result_sig, gt_de_result_sig)

        # 2, 3. Compute the log fold change Spearman correlation, direction agreement.
        logfc_results = logfc_metrics(pred_de_result, gt_de_result_sig)
        logfc_spearman_corr = logfc_results["logfc_spearman"]
        direction_agreement = logfc_results["direction_agreement"]

        de_metrics_by_group[condition] = {
            "deg_overlap": float(deg_overlap),
            "logfc_spearman": float(logfc_spearman_corr),
            "direction_agreement": float(direction_agreement),
        }

    return de_metrics_by_group

def condition_discrimination_score(
    preds_groups: Dict[str, np.ndarray],
    gt_groups: Dict[str, np.ndarray], 
    distance_measure: Literal["peusdo_bulk", "energy_distance", "sinkhorn"] = "peusdo_bulk"
) -> float:
    """
    Compute the condition discrimination score for each condition.
    CDS = 1 - r_t / T-1, r_t = number of incorrect conditions closer to the predicted condition than the correct one

    Args:
        preds_groups (Dict[str, np.ndarray]): A dictionary mapping condition to predicted expression values.
        gt_groups (Dict[str, np.ndarray]): A dictionary mapping condition to ground truth expression values.

    Returns:
        Dict[str, float]: A dictionary mapping condition to the condition discrimination score.
    """
    cds = {}
    
    conditions = list(preds_groups.keys())
    n = len(conditions)
    dists = np.zeros((n, n))

    if distance_measure == "pseudo_bulk":
        pred_pseudobulks = np.stack([
            preds_groups[c].mean(axis=0)
            for c in conditions
        ])

        gt_pseudobulks = np.stack([
            gt_groups[c].mean(axis=0)
            for c in conditions
        ])

        # [pred condition, GT condition]
        dists = pairwise_distances(
            pred_pseudobulks,
            gt_pseudobulks,
            metric="euclidean",
        )

    elif distance_measure == "energy_distance":
        for i, pred_cond in enumerate(conditions):
            for j, gt_cond in enumerate(conditions):
                dists[i, j] = energy_distance(
                    preds_groups[pred_cond],
                    gt_groups[gt_cond],
                )

    elif distance_measure == "sinkhorn":
        raise NotImplementedError("Sinkhorn distance measure is not implemented yet.")

    for idx, condition in enumerate(preds_groups):
        pred_dist = dists[idx, :]

        incorrect_closer_count = np.sum(pred_dist < pred_dist[idx])

        cds[condition] = 1 - incorrect_closer_count / (n - 1)

    return float(np.mean(list(cds.values())))

def _compute_group_pairwise_distances(
        groups: Dict[str, np.ndarray], 
        distance_measure: Literal["pseudo_bulk", "energy_distance", "sinkhorn"] = "pseudo_bulk"
) -> np.ndarray:
    """
    Compute pairwise distances between groups based on the specified distance measure.

    Args:
        groups (Dict[str, np.ndarray]): A dictionary mapping condition to expression values.
        distance_measure (Literal["pseudo_bulk", "energy_distance", "sinkhorn"]): The distance measure to use.

    Returns:
        np.ndarray: An array of pairwise distances between groups, of shape (n * (n - 1) / 2,), where n is the number of groups.
    """
    
    # Group conditions
    conditions = list(groups.keys())
    n = len(conditions)

    # Compute pairwise distances
    if distance_measure == "pseudo_bulk":
        pred_pseudobulks = np.stack([
            groups[c].mean(axis=0)
            for c in conditions
        ])

        dists = pdist(pred_pseudobulks, metric="euclidean") # (n * (n - 1) / 2,)

    elif distance_measure == "energy_distance":
        dists = []
        for i in range(n):
            for j in range(i+1, n):                
                dists.append(energy_distance(
                    groups[conditions[i]],
                    groups[conditions[j]],
                ))
        dists = np.array(dists)

    return dists

def global_distance_spearman(
    pred_groups: Dict[str, np.ndarray], 
    gt_groups: Dict[str, np.ndarray],
    distance_measure: Literal["pseudo_bulk", "energy_distance", "sinkhorn"] = "pseudo_bulk"
) -> Tuple[float, float, float, float]:
    """
    Compute the Spearman correlation of global distances between predicted and ground truth expression values.

    Args:
        pred_groups (Dict[str, np.ndarray]): A dictionary mapping condition to predicted expression values.
        gt_groups (Dict[str, np.ndarray]): A dictionary mapping condition to ground truth expression values.

    Returns:
        Tuple[float, float, float, float]: Spearman correlation, Spearman p-value, Pearson correlation, and
         Pearson p-value of global distances between predicted and ground truth expression values.
    """

    pred_dists = _compute_group_pairwise_distances(pred_groups, distance_measure=distance_measure)
    gt_dists = _compute_group_pairwise_distances(gt_groups, distance_measure=distance_measure)

    # Compute Spearman correlation
    spearman_corr, spearman_pval = spearmanr(pred_dists, gt_dists)

    # Compute Pearson correlation
    pearson_corr, pearson_pval = pearsonr(pred_dists, gt_dists)


    return spearman_corr, spearman_pval, pearson_corr, pearson_pval

def local_neighborhood_precision(
    pred_groups: Dict[str, np.ndarray],
    gt_groups: Dict[str, np.ndarray],
    distance_measure: Literal["pseudo_bulk", "energy_distance", "sinkhorn"] = "pseudo_bulk",
    k: int = 5,
) -> float:
    """
    Compute the local neighborhood precision for predicted neighborhood based on the ground truth.
    AP@k(c) denotes the average precision at k for condition c.
    AP@k(c) := k / number of conditions in predicted neighborhood within the radius that include ALL k nearest neighbors in the ground truth space,
    aka the fraction of predicted neighbors within the radius that are relevant according to the ground truth.

    Args:
        pred_groups (`Dict[str, np.ndarray]`): A dictionary mapping condition to predicted expression values.
        gt_groups (`Dict[str, np.ndarray]`): A dictionary mapping condition to ground truth expression values.
        distance_measure (`Literal["pseudo_bulk", "energy_distance", "sinkhorn"]`, optional): Distance measure to use. Defaults to "pseudo_bulk".
        k (`int`, optional): Number of nearest neighbors to consider. Defaults to 5.

    Returns:
        float: Local neighborhood precision.
    """
    conditions = list(pred_groups.keys())
    n = len(conditions)

    pred_dists = _compute_group_pairwise_distances(pred_groups, distance_measure=distance_measure) 
    gt_dists = _compute_group_pairwise_distances(gt_groups, distance_measure=distance_measure)
    pred_dists = squareform(pred_dists)
    gt_dists = squareform(gt_dists)

    scores = []
    k = min(k, n - 1)  # Ensure k does not exceed the number of available neighbors
    for i, _ in enumerate(conditions):
        other_idx = np.arange(n) != i
        candidates = np.arange(n)[other_idx]

        # k closest conditions in GT space
        gt_neighbors = candidates[
            np.argsort(gt_dists[i, other_idx])[:k]
        ]

        # Where is the furthest of those GT neighbors in predicted condition space?
        radius = pred_dists[i, gt_neighbors].max()

        # How many predicted conditions are inside that radius?
        n_inside = np.sum(
            pred_dists[i, other_idx] <= radius
        )

        scores.append(k / n_inside)

    return float(np.mean(scores))

def compute_structure_metrics(
    pred_groups: Dict[str, np.ndarray],
    gt_groups: Dict[str, np.ndarray],
    distance_measure="pseudo_bulk",
    k=5,
) -> Dict[str, float]:
    """
    Compute perturbation structure metrics for each condition.

    Args:
        pred_groups (Dict[str, np.ndarray]): A dictionary mapping condition to predicted expression values.
        gt_groups (Dict[str, np.ndarray]): A dictionary mapping condition to ground truth expression values.
        ctrl_adata (ad.AnnData): Control AnnData object.
        k (int, optional): Number of nearest neighbors to consider for mAP@k. Defaults to 5.

    Returns:
        Dict[str, float]: A dictionary mapping condition to the discrimination metric.
        - cds (float), [0,1]: Perturbation discrimination score for the condition.
        - global_distance_spearman (float), [-1,1]: Spearman correlation of global distances between predicted and ground truth expression values.
        - global_distance_spearman_pval (float): P-value of the Spearman correlation.
        - mAP@k (float): Mean average precision at k for the perturbation structure metric.
    """

    cds = condition_discrimination_score(pred_groups, gt_groups, distance_measure=distance_measure)

    global_dist_results = global_distance_spearman(pred_groups, gt_groups, distance_measure=distance_measure)
    global_dist_spearman, global_dist_spearman_pval, global_dist_pearson, global_dist_pearson_pval = global_dist_results

    map_at_k = local_neighborhood_precision(pred_groups, gt_groups, distance_measure=distance_measure, k=k)

    return {
        "cds": float(cds),
        "global_distance_spearman": float(global_dist_spearman),
        "global_distance_spearman_pval": float(global_dist_spearman_pval),
        "global_distance_pearson": float(global_dist_pearson),
        "global_distance_pearson_pval": float(global_dist_pearson_pval),
        "mAP@k": float(map_at_k),
    }


def compute_perturbation_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    perturbations: List[str],
    covariates: List[dict],
    ctrl_adata: ad.AnnData,
    fdr=0.05,
    neighborhood_k=5,
    top_deg=100,
):
    pred_groups, gt_groups = _group_results_by_condition(
        preds,
        gts,
        perturbations,
        covariates,
    )

    # Remove control groups 
    pred_groups = {cond: group for cond, group in pred_groups.items() if "control" not in cond}
    gt_groups = {cond: group for cond, group in gt_groups.items() if "control" not in cond}

    for cond, group in pred_groups.items():
        print(f"Condition: {cond}, shape of samples: {group.shape}\n")

    print(f"Number of conditions: {len(pred_groups)}")

    print(f"Computing expression fidelity metrics by group...")
    expression_metrics_by_group = compute_expression_fidelity_metrics_by_group(
        pred_groups,
        gt_groups,
        ctrl_adata,
        fdr=fdr,
        top_deg=top_deg
    )
    # Overall metrics averaged across conditions
    expression_metrics = {}
    for condition in expression_metrics_by_group:
        for k, v in expression_metrics_by_group[condition].items():
            if np.isnan(v):
                continue
            if k not in expression_metrics:
                expression_metrics[k] = []
            expression_metrics[k].append(v)
    for k in expression_metrics:
        expression_metrics[k] = np.mean(expression_metrics[k])

    print(f"Computing differential expression metrics by group...")
    de_metrics_by_group = compute_differential_expression_metrics_by_group(
        pred_groups,
        gt_groups,
        ctrl_adata,
        fdr=fdr,
    )
    # Overall DE metrics averaged across conditions
    de_metrics = {}
    for condition in de_metrics_by_group:
        for k, v in de_metrics_by_group[condition].items():
            if np.isnan(v):
                continue
            if k not in de_metrics:
                de_metrics[k] = []
            de_metrics[k].append(v)
    for k in de_metrics:
        de_metrics[k] = np.mean(de_metrics[k])

    print(f"Computing perturbation discrimination metrics by group...")
    discrimination_metrics = compute_structure_metrics(
        pred_groups,
        gt_groups,
        distance_measure="energy_distance",
        k=neighborhood_k,
    )

    return {
        "expression": expression_metrics,
        "differential_expression": de_metrics,
        "perturbation_discrimination": discrimination_metrics,
    }
