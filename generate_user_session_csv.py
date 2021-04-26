# utility for generating CSV and HTML view of selected user session, highlighting the suggestions positions and accepted text

import pandas as pd
import os
import ast
import datetime
import html
import sys
import string

SESSIONS_CSV = "./logs/all_sessions.csv"
today = datetime.datetime.today().strftime('%Y%m%d')

def convert_timestamp(timestamp):
    # try:
    #     # when timestamp is in seconds
    #     date = datetime.datetime.fromtimestamp(timestamp)
    # except (ValueError):
    # when timestamp is in miliseconds
    date = datetime.datetime.fromtimestamp(timestamp / 1000)
    return date

# generate csv and html files with session text and suggestions
def extract_user_suggestions(userid, filename):
    if os.path.exists(filename):
        try:
            df_all_sessions = pd.read_csv(filename)
        except (IOError, Exception) as e:
            print(e)
            return
    # get uesr's last session
    df_user = df_all_sessions[df_all_sessions.userid == userid]
    if len(df_user) == 0:
        print(f"user {userid} does not exist n file")
        return
    suggestions = df_user.iloc[-1, :].suggestions
    full_text = df_user.iloc[-1, :].text
    suggestions_list = ast.literal_eval(suggestions)
    output = pd.DataFrame(suggestions_list)
    output['timestamp'] = output['timestamp'].apply(convert_timestamp)
    # print(output.head(3))
    USER_SUGGESTIONS_FILE = "./logs/user_session/user_suggestions_" + str(userid) + "_" + today + ".csv"

    # write to file
    try:
        output.to_csv(USER_SUGGESTIONS_FILE, index=False)
        save_HTML(full_text, output, userid)
    except (PermissionError, Exception) as e:
        print(e)
        return


# find substring within a string (both in list representation)
# returns tuple (start idx, end idx) where substring exist in list.
# for example: find_sub_list(['my','name','is'], ['hello','my','name','is','bob'])
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    k = (i for i,e in enumerate(l) if e==sl[0])
    for ind in k:
    # for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    # error if sub string is not found
    assert len(results)>0, "failed to find substring: " + str(sl) + ' ' + str(l)
    # return the first substring found
    return results[0]

# Prevent special characters like & and < to cause the browser to display something other than what you intended.
def html_escape(text):
    return html.escape(text)

def remove_punctuation(s):
    punctuation = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in punctuation)
    return s

# get the word positions in text where suggestion was given (by the len of the input text of each request)
def get_suggest_positions(df):
    stops_pos = []
    for row_no, row in df.iterrows():
        stops_pos.append(len(row["input"].split())-1)
    return stops_pos[:-1]

# generate HTML highlighting the suggested points and the accepted text
def save_HTML(full_text, df, userid):
    USER_HTML_FILE = "./logs/user_session/user_suggestions_" + str(userid) + "_" + today + ".HTML"
    # add full text as the last line to df
    data = dict(userid = userid, timestamp= datetime.datetime.now(), input= full_text, accepted=-1)
    # df.loc[len(df.index)] = data
    df = df.append(data, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    # weight 1 is set to accepted word, O otherwise. Initialize all words to 0
    weights = []
    text = full_text.split()
    for i in range(len(text)):
        weights.append(0)

    # idx counts the current row in the full text
    idx = 0
    # handle each suggestion point (each row in df): if text accepted, set word's weight to 1.
    for row_no, row in df.iterrows():
        # delta is calculated from second row
        if row_no == 0:
            prev_row = row
            idx = len(row["input"].split()) - 1
            continue
        # if no option selected continue
        accepted_option = prev_row["accepted"]
        if accepted_option == -1:
            prev_row = row
            idx = len(row["input"].split()) - 1
            continue
        text1 = prev_row["input"]
        text2 = row["input"]
        # ignore punctuation when calculating text delta between 2 suggestion points
        delta = remove_punctuation(text2.replace(text1,''))
        opt = f'option{accepted_option}'
        option_text = remove_punctuation(prev_row[opt])
        # skip highlighting suggestion with punctuation only
        if option_text.strip() == "":
            prev_row = row
            idx = len(row["input"].split()) - 1
            continue
        # locate accepted words in text
        opt_idx = find_sub_list(option_text.split(), delta.split())
        start = idx + 1 + opt_idx[0]
        end = start + (opt_idx[1] - opt_idx[0])
        # set accepted words weight to 1 to highlight them
        for j in range(end+1):
            if j < start:
                continue
            weights[j] = 1

        idx = len(row["input"].split()) - 1
        prev_row = row

    # get all positions where suggestion was given
    stops = get_suggest_positions(df)
    highlighted_text = []
    for i, word in enumerate(text):
        weight = weights[i]
        if weight == 1:
            # set highlight for accepted word (= were set with weight 1)
            highlighted_text.append(
                '<span style="background-color:rgba(135,206,250,' + str(0.5) + ');">' + html_escape(word) + '</span>')
        else:
            highlighted_text.append(word)
        # highlight all positions where suggestion was given (accepted or not)
        if i in stops:
            t = f'_SG{stops.index(i)}_'
            highlighted_text.append(
                '<span class="tooltip" style="background-color:rgba(206,250,135,' + str(0.5) + ');">' + html_escape(t) + '</span>')

    highlighted_text = ' '.join(highlighted_text)

    # save details table to html
    df1 = df[["timestamp", "option1", "option2", "option3", "accepted"]][:-1]
    df1.to_html(USER_HTML_FILE)

    # add highlighted text to HTML
    Html_file = open(USER_HTML_FILE, "a", encoding="utf-8")
    t = "<br><br>" + highlighted_text
    Html_file.write(t)
    Html_file.close()


if __name__ == "__main__":
    # SET YOUR USER -------------
    userid = 1000
    # ---------------------------
    filename = SESSIONS_CSV
    # if used from cmd run for example: "python generate_user_session_csv.py {userid}"
    if len(sys.argv) > 1:
        userid = int(sys.argv[1])
        if len(sys.argv) > 2:
            filename = sys.argv[2]

    # print(userid)
    # print(filename)
    if not os.path.exists(filename):
        print("missing CSV")
    else:
        extract_user_suggestions(userid, filename)


