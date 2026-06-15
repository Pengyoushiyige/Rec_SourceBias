import os
import pandas as pd

def create_sample(src_dir, dest_dir, n_behaviors=15):
    print(f"Creating sample dataset from {src_dir} to {dest_dir}...")
    
    # 1. Create directory structure
    for sub in ['train', 'dev', 'text']:
        os.makedirs(os.path.join(dest_dir, sub), exist_ok=True)
        
    # 2. Sample behaviors files and collect referenced item/news IDs
    item_ids = set()
    
    # Files to process
    behavior_files = [
        ('train', 'behaviors.tsv'),
        ('train', 'behaviors1.tsv'),
        ('train', 'behaviors2.tsv'),
        ('dev', 'behaviors.tsv')
    ]
    
    for sub, filename in behavior_files:
        src_file = os.path.join(src_dir, sub, filename)
        dest_file = os.path.join(dest_dir, sub, filename)
        
        if not os.path.exists(src_file):
            print(f"Warning: behavior file {src_file} does not exist. Skipping.")
            continue
            
        # Read the first n_behaviors rows using pandas
        # Since it is a tsv without header, we parse it using '\t'
        df = pd.read_csv(src_file, sep='\t', header=None, nrows=n_behaviors)
        
        # Write to destination
        df.to_csv(dest_file, sep='\t', header=None, index=False)
        print(f"Sampled {len(df)} rows from {src_file} -> {dest_file}")
        
        # Collect item IDs
        # Column 3 (index 3) is clicked_news (space separated history)
        # Column 4 (index 4) is impressions (space separated candidates with -label suffix)
        for row in df.itertuples(index=False):
            # clicked_news
            clicked = row[3]
            if pd.notna(clicked) and isinstance(clicked, str):
                for item in clicked.split():
                    item_ids.add(item)
            # impressions
            impressions = row[4]
            if pd.notna(impressions) and isinstance(impressions, str):
                for imp in impressions.split():
                    item_id = imp.split('-')[0]
                    item_ids.add(item_id)
                    
    print(f"Collected {len(item_ids)} unique item IDs referenced in behaviors.")
    
    # 3. Filter all news files in the 'text' directory
    src_text_dir = os.path.join(src_dir, 'text')
    dest_text_dir = os.path.join(dest_dir, 'text')
    
    if os.path.exists(src_text_dir):
        for filename in os.listdir(src_text_dir):
            if not filename.endswith('.tsv'):
                continue
            src_file = os.path.join(src_text_dir, filename)
            dest_file = os.path.join(dest_text_dir, filename)
            
            # Read and determine format: parsed or raw
            with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            
            is_parsed = False
            if first_line.startswith('id\t') or first_line.strip().split('\t')[0] == 'id':
                is_parsed = True
                
            if is_parsed:
                # Parsed file has header and specific columns
                df_news = pd.read_csv(src_file, sep='\t')
                df_filtered = df_news[df_news['id'].astype(str).isin(item_ids)]
                df_filtered.to_csv(dest_file, sep='\t', index=False)
                print(f"Filtered parsed news {src_file} -> {dest_file} ({len(df_filtered)} rows)")
            else:
                # Raw file has no header, id in column 0
                df_news = pd.read_csv(src_file, sep='\t', header=None)
                df_filtered = df_news[df_news[0].astype(str).isin(item_ids)]
                df_filtered.to_csv(dest_file, sep='\t', header=None, index=False)
                print(f"Filtered raw news {src_file} -> {dest_file} ({len(df_filtered)} rows)")

if __name__ == "__main__":
    import argparse
    
    # Compute base dir (root of repository) based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    default_src = os.path.join(base_dir, "dataset", "Amazon_Beauty")
    default_dest = os.path.join(base_dir, "dataset", "Amazon_Beauty_sample")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=default_src, help="Source dataset path")
    parser.add_argument("--dest", type=str, default=default_dest, help="Target sample dataset path")
    parser.add_argument("--n", type=int, default=15, help="Number of rows to sample from behaviors")
    args = parser.parse_args()
    
    create_sample(args.src, args.dest, args.n)
