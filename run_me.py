import json
import pathlib
import typing as tp
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import final_solution

# import final_solution.solution_stupid


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)



def main():
    texts = load_data('data/test_texts.json')
    nameAndSynonyms = pd.read_excel('company/names and synonyms_upd (3).xlsx')
    issuerIdAndSynonyms = {}
    for i, n in nameAndSynonyms.iterrows():
        substingsArray = [x.strip() for x in n['Company_Ident_Text'].split(',') if x]
        issuerIdAndSynonyms.update({
            n['issuerid']: substingsArray
        })
    vectorizer_loaded = pickle.load(open("models/vectorizer.sav", 'rb'))

    loaded_model = pickle.load(open("models/log_reg.sav", 'rb'))

    scores = final_solution.solution.score_texts(texts, issuerIdAndSynonyms, loaded_model, vectorizer_loaded)
    save_data(scores, "results/output_result.json")


if __name__ == '__main__':
    main()
