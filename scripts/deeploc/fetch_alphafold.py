import argparse
import os
import requests
import pandas as pd
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_uniprot_mapping_batch(ids, from_db="Ensembl_Protein"):
    """Map a small batch of IDs to UniProtKB Accession IDs."""
    if not ids:
        return {}
    
    submit_url = "https://rest.uniprot.org/idmapping/run"
    try:
        response = requests.post(
            submit_url,
            data={
                "from": from_db,
                "to": "UniProtKB",
                "ids": ",".join(ids)
            },
            timeout=30
        )
        response.raise_for_status()
        job_id = response.json()["jobId"]
        
        # Poll for completion
        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        while True:
            status_resp = requests.get(status_url, timeout=30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            
            if "results" in status_data or status_data.get("jobStatus") == "FINISHED":
                break
            time.sleep(1)
            
        # Get results
        results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
        results_resp = requests.get(results_url, timeout=30)
        results_resp.raise_for_status()
        results_data = results_resp.json()
        
        mapping = {}
        for item in results_data.get("results", []):
            f_id = item.get("from")
            t_obj = item.get("to")
            if not f_id or not t_obj:
                continue
            
            if isinstance(t_obj, dict):
                t_id = t_obj.get("primaryAccession") or t_obj.get("uniProtkbId") or str(t_obj)
            else:
                t_id = str(t_obj)
            mapping[f_id] = t_id
        return mapping
    except Exception as e:
        print(f"Error mapping batch from {from_db}: {e}")
        return {}

def get_ensembl_info_bulk(ids):
    """Retrieve Ensembl hierarchy (Translation -> Transcript -> Gene) via bulk lookup."""
    if not ids:
        return {}
    
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {"ids": ids}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in Ensembl bulk lookup: {e}")
        return {}

def get_uniprot_mapping(ids, cache_path="data/id_mapping_cache.json"):
    """Robustly map IDs using a 4-stage hierarchical bridge and local caching."""
    mapping = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            mapping = json.load(f)
            
    remaining_ids = [idx for idx in ids if idx not in mapping]
    if not remaining_ids:
        return mapping

    # Stage 1: Direct ENSP -> UniProtKB
    print(f"Stage 1: Mapping {len(remaining_ids)} ENSP IDs directly...")
    batch_size = 200
    for i in range(0, len(remaining_ids), batch_size):
        batch = remaining_ids[i : i + batch_size]
        mapping.update(get_uniprot_mapping_batch(batch, "Ensembl_Protein"))
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(mapping, f)
        
    failed_ensp = [idx for idx in remaining_ids if idx not in mapping]
    if not failed_ensp:
        return mapping
    
    # Stage 2 & 3: Bridge via ENST and ENSG
    print(f"Stage 2: Resolving Genes for {len(failed_ensp)} unresolved IDs...")
    ensp_to_ensg = {}
    # First, lookup ENST parents for ENSP
    ensembl_results = {}
    for i in range(0, len(failed_ensp), batch_size):
        batch = failed_ensp[i : i + batch_size]
        ensembl_results.update(get_ensembl_info_bulk(batch))
    
    enst_to_ensp = {}
    all_enst = []
    for ensp, info in ensembl_results.items():
        if info and "Parent" in info:
            enst = info["Parent"]
            enst_to_ensp[enst] = ensp
            all_enst.append(enst)
            
    # Then, lookup ENSG parents for ENST
    print(f"Stage 3: Fetching Gene IDs for {len(all_enst)} transcripts...")
    enst_results = {}
    for i in range(0, len(all_enst), batch_size):
        batch = all_enst[i : i + batch_size]
        enst_results.update(get_ensembl_info_bulk(batch))
        
    ensg_to_ensp_list = {} # One Gene can have multiple ENSPs
    all_ensg = []
    for enst, info in enst_results.items():
        if info and "Parent" in info:
            ensg = info["Parent"]
            ensp = enst_to_ensp[enst]
            if ensg not in ensg_to_ensp_list:
                ensg_to_ensp_list[ensg] = []
            ensg_to_ensp_list[ensg].append(ensp)
            all_ensg.append(ensg)
            
    # Stage 4: Map ENSG -> UniProtKB
    print(f"Stage 4: Mapping {len(all_ensg)} Gene IDs to UniProtKB...")
    unique_ensg = list(set(all_ensg))
    ensg_to_uniprot = {}
    for i in range(0, len(unique_ensg), batch_size):
        batch = unique_ensg[i : i + batch_size]
        ensg_to_uniprot.update(get_uniprot_mapping_batch(batch, "Ensembl"))
        
    # Final Merge
    for ensg, uids in ensg_to_uniprot.items():
        if ensg in ensg_to_ensp_list:
            for ensp in ensg_to_ensp_list[ensg]:
                mapping[ensp] = uids
                
    # Final Cache Save
    with open(cache_path, "w") as f:
        json.dump(mapping, f)
        
    print(f"Final mapping coverage: {len(mapping)}/{len(ids)} ({len(mapping)/len(ids)*100:.1f}%)")
    return mapping

def download_cif(original_id, fetch_id, dest_dir):
    """Download the AlphaFold CIF file."""
    dest = os.path.join(dest_dir, f"{original_id}.cif")
    if os.path.exists(dest):
        return True
    
    if not fetch_id:
        return False
        
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{fetch_id}"
    try:
        req = requests.get(api_url, timeout=15)
        if req.status_code == 200:
            data = req.json()
            if data:
                cif_url = data[0].get("cifUrl")
                if cif_url:
                    cif_req = requests.get(cif_url, timeout=30)
                    if cif_req.status_code == 200:
                        with open(dest, "wb") as f:
                            f.write(cif_req.content)
                        return True
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Fetch AlphaFold structures with 100% target coverage.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV.")
    parser.add_argument("--out", type=str, default="data/alphafold_cif", help="Output directory.")
    parser.add_argument("--threads", type=int, default=16, help="Threads.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    id_col = "sid" if "sid" in df.columns else ("ACC" if "ACC" in df.columns else df.columns[0])
    ids = df[id_col].unique().tolist()
    
    is_ensembl = any(str(i).startswith("ENSP") for i in ids[:20])
    mapping = get_uniprot_mapping(ids) if is_ensembl else {i: i for i in ids}
    
    print(f"Starting download for {len(ids)} structures...")
    success = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_cif, oid, mapping.get(oid), args.out): oid for oid in ids}
        for future in tqdm(as_completed(futures), total=len(ids)):
            if future.result():
                success += 1

    print(f"DONE. Success: {success}/{len(ids)} ({success/len(ids)*100:.1f}%)")

if __name__ == "__main__":
    main()
