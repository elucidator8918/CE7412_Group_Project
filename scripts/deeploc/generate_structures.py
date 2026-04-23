import os
import sys
import pandas as pd
import logging
import requests
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )

def query_esmfold(sequence: str, retries=3):
    """Query the Meta ESMFold API with retries."""
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    if len(sequence) > 400:
        sequence = sequence[:400]
        
    for attempt in range(retries):
        try:
            response = requests.post(url, data=sequence, timeout=45)
            if response.status_code == 200:
                return response.text
            elif response.status_code in [502, 504]:
                time.sleep(2 * (attempt + 1)) # Exponential backoff
                continue
            else:
                return None
        except Exception:
            time.sleep(1)
            continue
    return None

def process_protein(row, output_dir, sleep=0.1):
    prot_id = row["prot_id"]
    sequence = row["sequence"]
    output_path = output_dir / f"{prot_id}.pdb"
    
    if output_path.exists():
        return True
        
    pdb_content = query_esmfold(sequence)
    if pdb_content:
        with open(output_path, "w") as f:
            f.write(pdb_content)
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of proteins to process")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel API workers")
    args = parser.parse_args()
    
    setup_logging()
    
    data_root = Path("./deeploc_data")
    output_dir = data_root / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_paths = [
        data_root / "Swissprot_Train_Validation_dataset.csv",
        data_root / "hpa_testset.csv"
    ]
    
    all_data = []
    for p in csv_paths:
        if p.exists():
            tmp_df = pd.read_csv(p)
            id_col = "ACC" if "ACC" in tmp_df.columns else ("ACC\n" if "ACC\n" in tmp_df.columns else "sid")
            seq_col = "Sequence" if "Sequence" in tmp_df.columns else "fasta"
            all_data.append(tmp_df[[id_col, seq_col]].rename(columns={id_col: "prot_id", seq_col: "sequence"}))

    df = pd.concat(all_data).drop_duplicates(subset=["prot_id"])
    existing = {f.stem for f in output_dir.glob("*.pdb")}
    to_process = df[~df["prot_id"].isin(existing)]
    
    if args.limit:
        to_process = to_process.head(args.limit)
    
    if len(to_process) == 0:
        logger.info("All structures already generated.")
        return

    logger.info(f"To generate: {len(to_process)} using {args.workers} workers.")
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_protein, row, output_dir): row for _, row in to_process.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

    logger.info("Generation complete.")

if __name__ == "__main__":
    main()
