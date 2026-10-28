import mygene


def add_gene_symbol_and_ensembl(adata, organism: str = "human"):
    print(f"Making sure both gene symbols and ensembl ids exist ...")

    gene_ensembl_exists = False
    gene_symbol_exists = False
    if "gene_ensembl" in adata.var.columns:
        gene_ensembl_exists = True
    if "gene_symbol" in adata.var.columns:
        gene_symbol_exists = True

    if not gene_ensembl_exists and not gene_symbol_exists:
        adata.var_names_make_unique()
        if adata.var_names[0].startswith("ENS"):
            adata.var['gene_ensembl'] = adata.var_names
            gene_lists = adata.var["gene_ensembl"].tolist()
            ensembl_to_symbol = convert_ensembl_to_symbol(gene_lists, organism=organism)
            adata.var["gene_symbol"] = adata.var["gene_ensembl"].map(ensembl_to_symbol)
        else:
            adata.var['gene_symbol'] = adata.var_names
            gene_lists = adata.var["gene_symbol"].tolist()
            symbol_to_ensembl = convert_symbol_to_ensembl(gene_lists, organism=organism)
            adata.var["gene_ensembl"] = adata.var["gene_symbol"].map(symbol_to_ensembl)
        gene_ensembl_exists = True
        gene_symbol_exists = True
    elif gene_ensembl_exists and not gene_symbol_exists:
        gene_lists = adata.var["gene_ensembl"].tolist()
        ensembl_to_symbol = convert_ensembl_to_symbol(gene_lists, organism=organism)
        adata.var["gene_symbol"] = adata.var["gene_ensembl"].map(ensembl_to_symbol)
        gene_symbol_exists = True
    elif gene_symbol_exists and not gene_ensembl_exists:
        gene_lists = adata.var["gene_symbol"].tolist()
        symbol_to_ensembl = convert_symbol_to_ensembl(gene_lists, organism=organism)
        adata.var["gene_ensembl"] = adata.var["gene_symbol"].map(symbol_to_ensembl)
        gene_ensembl_exists = True
    assert gene_ensembl_exists and gene_symbol_exists, "Both gene_ensembl and gene_symbol must exist in adata.var"

    print(f"Finished adding gene symbols and ensembl ids. Gene symbol exists: {gene_symbol_exists}, Gene ensembl exists: {gene_ensembl_exists}")

    return adata

def convert_ensembl_to_symbol(ensembl_ids: list[str], 
                              organism: str = "human") -> dict[str, str | None]:
    # Strip version suffix (e.g., "ENSG00000141510.16" -> "ENSG00000141510")
    # Keep mapping to restore original requested strings in the returned dict
    clean_to_raw = {raw_id.split(".")[0].strip(): raw_id for raw_id in ensembl_ids}

    mg = mygene.MyGeneInfo()
    query_result = mg.querymany(list(clean_to_raw.keys()), 
                                scopes="ensembl.gene", 
                                fields="symbol", 
                                species=organism,
                                as_dataframe=False,
                                returnall=False)
    mapping = {}
    for item in query_result:
        query = item["query"]
        symbol = item.get("symbol", None)
        mapping[clean_to_raw[query]] = symbol

    return mapping
    
def convert_symbol_to_ensembl(symbols: list[str], 
                              organism: str = "human") -> dict[str, str | None]:
    # Extract mapping (handles cases where ensembl is a list or dict)
    def _get_ensembl_id(val):
        if isinstance(val, dict):
            return val.get("gene")
        elif isinstance(val, list) and len(val) > 0:
            return val[0].get("gene")
        return None
    
    mg = mygene.MyGeneInfo()
    query_result = mg.querymany(symbols, 
                                scopes="symbol", 
                                fields="ensembl.gene", 
                                species=organism,
                                as_dataframe=False,
                                returnall=False)
    mapping = {}
    for item in query_result:
        query = item["query"]
        ensembl_id = _get_ensembl_id(item.get("ensembl", None))
        mapping[query] = ensembl_id

    return mapping
