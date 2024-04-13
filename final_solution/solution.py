import typing as tp
import pandas as pd

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

def process_text(text, company_names):
    processed_text = text

    index = processed_text.find(company_names)
    end_index = index + 150
    if end_index > len(text):
        end_index = len(text)

    return processed_text[index:end_index]



def score_texts(
    messages: tp.Iterable[str],
    issuerIdAndSynonyms, loaded_model, vectorizer_loaded
) -> tp.Iterable[MessageResultType]:

    resultIssuerId = {'items': []}
    messageId = 0
    resultArray = []
    for text in messages:
        resultArray.append([])
        for key, subStrings in issuerIdAndSynonyms.items():
            for subString in subStrings:
                resString = subString
                if len(resString) < 4:
                    resString = " " + resString + " "
                if resString.lower() in text.lower():
                    resultIssuerId['items'].append({
                        'counterId': messageId,
                        'issuerid': key,
                        'synonym': subString.lower(),
                        'Company_Sentences': process_text(text.lower(), subString.lower()),
                    })
                    break
        messageId+=1
    df = pd.DataFrame(resultIssuerId['items'])

    X = df['Company_Sentences']

    X_test_vec = vectorizer_loaded.transform(X)
    y_pred = loaded_model.predict(X_test_vec)

    z = df['synonym']
    for i in range(0, len(y_pred)):
        resultArray[resultIssuerId['items'][i]['counterId']].append([resultIssuerId['items'][i]['issuerid'], float(y_pred[i])])
    return resultArray
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    # >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    # """
    # raise NotImplementedError
