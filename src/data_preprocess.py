from os import path
from pathlib import Path
from tqdm import tqdm

import swifter
import utils
import logging
import numpy as np
import pandas as pd
from parameters import parse_args
from transformers import AutoTokenizer

utils.setuplogger()
args = parse_args() # 解析配置参数
tokenizer = AutoTokenizer.from_pretrained(args.plm_model) # 加载预训练模型的 tokenizer
logging.info(f"args: {args}") # 日志记录参数配置
logging.info("Process data for training ") # 日志记录数据处理信息

def parse_news(source, target): #将原始文本数据转化为PLM能够理解的Token序列，并保存为下游训练/测试所用的格式
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    logging.info(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4],
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract'
                         ])
    news = news[['id', 'title', 'abstract']] #过滤多余字段
    news.fillna(' ', inplace=True) #填充缺失值

    def parse_row(row): #对单行数据（即单个商品/新闻）的文本摘要进行清洗、分词、编码，并统一输出格式
        abstract = row['abstract'].lower() #将商品的摘要文本转换为小写
        abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
            pad_to_max_length=True, truncation=True)
        if "bert-base-uncased" in args.plm_model:
            new_row = [
                row.id, abstract['input_ids'], abstract['token_type_ids'], abstract['attention_mask']
            ]
        elif "roberta-base" in args.plm_model:
            new_row = [
                row.id, abstract['input_ids'], [0] * len(abstract['input_ids']), abstract['attention_mask']
            ]
        return pd.Series(new_row,
                        index=[
                            'id', 'abstract_input_ids', 'abstract_token_type_ids', 'abstract_attention_mask'
                        ])

    parsed_news = news.swifter.apply(parse_row, axis=1) #按行循环处理news表格吧每一行数据送进parse_now函数中去进行tokenize分词
    parsed_news.to_csv(target, sep='\t', index=False) #保存为表格

    

if __name__ == "__main__":
    parse_news(path.join(args.text_source),
               path.join(args.text_parsed_target))