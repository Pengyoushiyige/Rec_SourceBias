import argparse

import utils
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="test",
                        choices=['train', 'test', 'loop'])
    parser.add_argument("--dataset", type=str, default='Amazon_Beauty')
    
    parser.add_argument("--model_type", type=str, default="BERT4Rec")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=100)

    # model training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--news_attributes", type=str, nargs='+', default=['abstract'])
       
    parser.add_argument("--num_words_abstract", type=int, default=100)
    parser.add_argument("--user_log_length", type=int, default=10)
    parser.add_argument("--word_embedding_dim", type=int, default=768)

    parser.add_argument("--user_log_mask", action="store_true", default=True)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=100000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument("--load_ckpt_name", type=str, default="epoch-3.pt")
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument("--plm_model", type=str, default="bert-base-uncased")
    parser.add_argument("--llm_model", type=str, default="llama")
    
    parser.add_argument("--negative_sampling_ratio", default=1, type=int)

    # dataset
    parser.add_argument("--behaviors_file", default="behaviors.tsv", type=str)
    parser.add_argument("--behaviors_file1", default="behaviors1.tsv", type=str)
    parser.add_argument("--behaviors_file2", default="behaviors2.tsv", type=str)
    parser.add_argument("--test_behaviors_file", default="behaviors.tsv", type=str)
    parser.add_argument("--human_news_file", default=None, type=str)
    parser.add_argument("--HGC_file", default=None, type=str)
    parser.add_argument("--llm_news_file", default="news_llama_parsed_bert-base-uncased.tsv", type=str)
    parser.add_argument("--llm_rewirte_news_file", default="news_llama_rewrite_bert-base-uncased.tsv", type=str)
    parser.add_argument("--seed", type=int, default=2023)
    
    ## debias
    parser.add_argument("--debias", action="store_true", default=False)
    parser.add_argument("--debias_type", type=str, default="emb_entropy", choices=["emb_entropy", "dis_user", "dis_news", "dis_double"])
    
    parser.add_argument("--disturb_ratio", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=-1)
    parser.add_argument("--loop_epochs", type=int, default=10)

    parser.add_argument("--user_loss", type=float, default=0.01)
    parser.add_argument("--news_loss", type=float, default=1)
    parser.add_argument("--test_epoch", type=int, default=3)
    
    ## dataset
    parser.add_argument("--text_source", type=str, default="")
    parser.add_argument("--text_parsed_target", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="")
    
    args = parser.parse_args()

    if args.human_news_file is None and args.HGC_file is not None:
        args.human_news_file = args.HGC_file
    elif args.HGC_file is None and args.human_news_file is not None:
        args.HGC_file = args.human_news_file
    elif args.human_news_file is None and args.HGC_file is None:
        args.human_news_file = "news_parsed_bert-base-uncased.tsv"
        args.HGC_file = "news_parsed_bert-base-uncased.tsv"

    import os
    dataset_dir = os.path.join("dataset", args.dataset)
    for attr in ['human_news_file', 'HGC_file', 'llm_news_file', 'llm_rewirte_news_file']:
        val = getattr(args, attr)
        if val and not os.path.isabs(val) and not val.startswith("dataset/"):
            alt_path = os.path.join(dataset_dir, "text", val)
            if not os.path.exists(val) or val in ["news_parsed_bert-base-uncased.tsv", "news_llama_parsed_bert-base-uncased.tsv", "news_llama_rewrite_bert-base-uncased.tsv"]:
                if os.path.exists(alt_path) or not os.path.exists(val):
                    setattr(args, attr, alt_path)

    for attr in ['behaviors_file', 'behaviors_file1', 'behaviors_file2', 'test_behaviors_file']:
        val = getattr(args, attr)
        if val and not os.path.isabs(val) and not val.startswith("dataset/"):
            subdirs = ["dev", "train"] if "test" in attr else ["train", "dev"]
            resolved = False
            for sd in subdirs:
                alt_path = os.path.join(dataset_dir, sd, val)
                if os.path.exists(alt_path):
                    setattr(args, attr, alt_path)
                    resolved = True
                    break
            if not resolved and not os.path.exists(val):
                setattr(args, attr, os.path.join(dataset_dir, "train", val))

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
