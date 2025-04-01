import ast

from aac_metrics.functional import spider
from bert_score import BERTScorer
from tqdm import tqdm
import pandas as pd
import statistics
import torch
import os

caption_data = [
    f"caption_data{os.path.sep}prompt_eng{os.path.sep}zero-shot_artpedia-captions.csv",
    f"caption_data{os.path.sep}prompt_eng{os.path.sep}one-shot_artpedia-captions.csv",
    f"caption_data{os.path.sep}prompt_eng{os.path.sep}few-shot_artpedia-captions.csv",
    f"caption_data{os.path.sep}prompt_eng{os.path.sep}multimodal-cot_artpedia-captions.csv",
    f"caption_data{os.path.sep}prompt_eng{os.path.sep}self-consistency_artpedia-captions.csv"
]

simple_df = pd.DataFrame(columns=["technique", "BERTscore",
                                   "SPICE", "CIDEr-D", "SPIDEr"])

detailed_df = simple_df.copy()


with (tqdm(desc="Computing metrics", total=len(caption_data)) as pbar):
    bert = BERTScorer(model_type="distilbert-base-uncased", lang="en",
                      device="cuda" if torch.cuda.is_available() else "cpu")

    for path in caption_data:
        technique_nm = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path)

        generated_captions = df["caption"].tolist()
        reference_captions = [ast.literal_eval(ref) for ref in df["visual_sentences"].tolist()]

        # BERTscore
        P, R, F1 = bert.score(generated_captions, reference_captions)
        df["BERTscore"] = F1

        # CIDEr-D, SPICE, SPIDEr
        score, scores = spider(generated_captions, reference_captions)

        df["SPICE"] = scores['spice'].tolist()
        df["CIDEr-D"] = scores['cider_d'].tolist()
        df["SPIDEr"] = scores['spider'].tolist()

        simple_row = {"technique": technique_nm,
               "BERTscore": statistics.mean(df["BERTscore"].tolist()),
               "SPICE": score['spice'].item(),
               "CIDEr-D": score['cider_d'].item(),
               "SPIDEr": score['spider'].item()
               }

        simple_df = pd.concat([simple_df, pd.DataFrame([simple_row])], ignore_index=True)

        df.to_csv(path, index=False)

        pbar.update(1)

    simple_df.to_csv(f"experiment_results{os.path.sep}prompt_eng{os.path.sep}simple_results.json")

with (tqdm(desc="Analyzing metrics", total=len(caption_data)) as pbar):
    for path in caption_data:
        df = pd.read_csv(path)
        technique_nm = os.path.basename(path).split("_")[0]

        bert_scores, spice_scores, cider_scores, spider_scores = \
            df["BERTscore"].tolist(), df["SPICE"].tolist(), df["CIDEr-D"].tolist(), df["SPIDEr"].tolist()

        row = {"technique": technique_nm,
               "BERTscore": {"MEAN": simple_df.loc[simple_df['technique'] == technique_nm, 'BERTscore'].values[0],
                             "STD": statistics.stdev(bert_scores),
                             "MAX": max(bert_scores),
                             "MIN": min(bert_scores),
                             },

               "SPICE": {"MEAN": simple_df.loc[simple_df['technique'] == technique_nm, 'SPICE'].values[0],
                         "STD": statistics.stdev(spice_scores),
                         "MAX": max(spice_scores),
                         "MIN": min(spice_scores),
                         },

               "CIDEr-D": {"MEAN": simple_df.loc[simple_df['technique'] == technique_nm, 'CIDEr-D'].values[0],
                           "STD": statistics.stdev(cider_scores),
                           "MAX": max(cider_scores),
                           "MIN": min(cider_scores),
                           },

               "SPIDEr": {"MEAN": simple_df.loc[simple_df['technique'] == technique_nm, 'SPIDEr'].values[0],
                          "STD": statistics.stdev(spider_scores),
                          "MAX": max(spider_scores),
                          "MIN": min(spider_scores),
                          }
               }

        detailed_df = pd.concat([detailed_df, pd.DataFrame([row])], ignore_index=True)
        pbar.update(1)

    detailed_df.to_csv(f'experiment_results{os.path.sep}prompt_eng{os.path.sep}detailed_results.csv',
                       index=False)
