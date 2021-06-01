"""
utility for comparing different models performance: next words suggestions, runtime and perplexity
"""
import pandas as pd
import os
import datetime
import sys
import re
from autocompleter import Autocompleter
from generate_user_session_csv import save_HTML
import math
import torch
from time import time

today = datetime.datetime.today().strftime('%Y%m%d')

def score_perplexity(model, tokenizer, docs):
    """
    calculate transformer perplexity for given documents
    :param model: language model (like GPT2,..)
    :param tokenizer: LM tokenizer
    :param docs: list of text documents
    :return: list of perplexity scores given by the models
    """

    scores = []
    model.eval()

    for doc in docs:
        tokenize_input = tokenizer.tokenize(doc)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        scores.append(math.exp(loss.loss))

    return scores

def compare_models(models, model_names, docs, docs_ids, performance=False):
    """
    compare doc's perplexity for given models
    If performance is True calculate next phrase prediction runtime on full doc text
    Results are saved to CSV
    :param models: LM
    :param model_names:  names list
    :param docs: documents list
    :param docs_ids: documents id list
    :param performance: T/F indicator if to calc the prediction runtime for next doc's words
    :return: DataFrame (results)
    """

    # output df
    cols = ['doc_id','doc_length', 'text']
    cols = cols + model_names
    if performance:
        for n in model_names:
            cols.append(n+ "_runtime")
    df_scores = pd.DataFrame(columns = cols)
    length, text, runtime = [], [], []
    for d in docs:
        length.append(len(d.split()))
        text.append(d[:15])
    df_scores['doc_id'] = docs_ids
    df_scores['doc_length'] = length
    df_scores['text'] = text

    # for each model calculate docs perplexity
    for i, completer in enumerate(models):
        # start = time()
        # print(score_perplexity(completer.model, completer.tokenizer, [text1]))
        # timing("---------  Scoring time {} -----------".format(names[i]), start)
        df_scores[model_names[i]] = score_perplexity(completer.model, completer.tokenizer, docs)
        # if performance check is needed - calculate prediction runtime for doc's next phrase
        if performance:
            runtime = []
            for d in docs:
                # check runtime of full document next words predicion
                start = time()
                completions, delta, options = completer.predict(d)
                runtime.append(time()-start)
            df_scores[completer.model_type + "_runtime"] = runtime

    df_scores.to_csv("./logs/user_session/compare_models"+"_"+today+".csv")
    return df_scores

def predict_doc(doc, doc_id, completer):
    """
    simulate text completion in the given document:
        - split doc to phrases. truncate last 3 words before punctuation if phrase is loner than 5
        - for each truncated phrase predict next words
    save predictions to CSV/HTML files
    :param doc:
    :param doc_id:
    :param completer: LM
    :return: result text, DF (words predictions)
    """

    # before splitting to phrases by punctuation ignore ".com" suffix: change "xx.com" to xxcom"
    doc = doc.replace(".com", "com")
    # split to phrases
    no_punc = re.split('[?.,:;]', doc)
    if no_punc[-1].strip() == "":
        no_punc = no_punc[:-1]

    # list of punctuation in the doc (will be added to text after prediction
    punc = []
    for i in doc:
        if i in ['?', '.', ',', ':', ";"]:
            punc.append(i)
    assert len(no_punc) >= len(punc)

    # list of truncated phrases from doc
    truncated = []
    # ground truth - the original text that was truncated
    GT =[]
    for t in no_punc:
        if len(t.split()) > 5:
            t = t.rstrip()
            truncated_text = t.rsplit(' ', 3)[0]
            # save ground truth that was truncated
            GT.append(t.replace(truncated_text, ''))
            truncated.append(truncated_text)
        else:
            GT.append("---")
            truncated.append(t)

    session_text = ""
    df = pd.DataFrame(columns=['accepted','input','option1', 'option2','option3','timestamp','userid','ground_truth'])

    # for each truncated phrase
    for i, t in enumerate(truncated):
        if len(t.split()) == len(no_punc[i].split()):
            # if truncated text equal to original text, no prediction is needed
            session_text = session_text + no_punc[i]
        else:
            # predict next words for truncated text
            session_text_in = session_text + truncated[i]
            _, _, options = completer.predict(session_text_in)
            # update session_text with the first option. add space if needed
            if session_text_in.endswith(" ") or options[0].startswith(" "):
                session_text = session_text_in + options[0]
            else:
                session_text = session_text_in + " " + options[0]
            assert i <= len(GT) -1 , "len GT/i: " + str(len(GT)) + '/' + str(i)
            # add suggestions to DF
            data = dict(userid='XXX', timestamp=datetime.datetime.now(), input= session_text_in, accepted=1, option1=options[0], option2=options[1], option3=options[2], ground_truth= GT[i])
            df = df.append(data, ignore_index=True)

        # add the original punctuation
        if i <= len(punc) - 1:
            session_text = session_text + punc[i]

    # save results to HTML
    # remove "/" from model name
    mdl_type = completer.model_type
    model_name  = mdl_type.replace("/", "")
    save_HTML(session_text, df, userid='TESTDOC_'+str(doc_id)+" "+model_name, mode="test")
    return session_text, df


def main():
    """
    run simulation for given docs
    COMPARE - check for each doc/model PPL (perplexity) and prediction runtime. save to CSV
    SUGGEST - for each doc truncate sentence endings and suggest words for it. saved to HTML
    :return:none
    """

    if len(doc_list)==0:
        # if no docs providedd - read from file
        if not os.path.exists(filename):
            print("missing CSV")
            return
        else:
            df_test_docs = pd.read_csv(filename)
            docs = df_test_docs["text"].to_list()
            docs_ids = df_test_docs["ID"].to_list()
            # if input includes a document id, get the document from the CSV
            if len(doc_id_list) > 0:
                # docs_ids = df_test_docs.loc[df_test_docs.ID == int(docid), 'ID'].to_list()
                # docs = df_test_docs.loc[df_test_docs.ID == int(docid), 'text'].to_list()
                docs, docs_ids = [], []
                for docid in docs_id_list:
                    doc = df_test_docs.loc[df_test_docs.ID == int(docid), 'text'].values[0]
                    docs.append(doc)
                    docs_ids.append(docid)
    else:
        docs = doc_list
        docs_ids = doc_id_list

    completers = []
    for m in model_types:
        completers.append(Autocompleter(m))

    if 'SUGGEST' in test:
        # simulate doc prediction
        for doc, doc_id in zip(docs, docs_ids):
            session_txts = []
            dfs = []
            for i, c in enumerate(completers):
                session_text, df = predict_doc(doc, doc_id, c)
                if i!=0:
                    # rename option1 column name for second and third models
                    df.rename(columns={"option1":"model"+str(i+1)+"_option1"}, inplace=True)
                session_txts.append(session_text)
                dfs.append(df)
            # merge dfs
            for i in range(len(dfs)):
                if i==0:
                    df_merge = dfs[0]
                else:
                    df_merge = pd.concat([df_merge, dfs[i]["model"+str(i+1)+"_option1"]], axis=1)
            save_HTML(session_txts[0], df_merge, userid='COMPARE_TESTDOC_' + str(doc_id), mode='compare')

    if 'COMPARE' in test:
        # compare models PPL. if performance=True check also suggestion runtime for full text
        compare_models(completers, model_types, docs, docs_ids, performance=True)

if __name__ == "__main__":
    # CMD: "python test_docs.py {filename} {doc_id_list}"

    # manual documents setup
    # doc_list = ['there is a book on the desk',
    #      'there is a plane on the desk',
    #      'there is a book in the desk']
    # doc_id_list = [1,2,3]

    doc_list, doc_id_list = [], []
    filename = "TEST_DOCS.csv"
    # doc_id_list =[14729632]

    test = ['SUGGEST', 'COMPARE']
    # test = ['COMPARE']
    model_types = ["distilgpt2", "fine_tuned", "gpt2", "openai-gpt"]
    # model_types = ["distilgpt2", "fine_tuned", "gpt2", "EleutherAI/gpt-neo-125M"]

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        docs_id_list = sys.argv[2]

    main()



#-----------------------------------------------------------------
    # model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    # model.eval()
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    # print(score_perplexity(model, tokenizer, a))