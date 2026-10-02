import mygene


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
