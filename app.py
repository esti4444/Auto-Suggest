from autocompleter import Autocompleter
from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import os
import datetime

# max number of returned words in option
MAX_LEN = 3
SESSIONS_CSV = "./logs/all_sessions.csv"

app = Flask(__name__)

# Initiate HTML
@app.route('/', methods=['POST','GET'])
def index():
    if not request.method == 'POST':
        return render_template('index.html')

# process prediction request
@app.route('/autocomplete', methods=['POST'])
def process():
    print("process request")
    data = request.json
    text = data.get('context', None)
    userid = data.get('userid', None)
    if text and userid:
        completions, delta, options = completer.predict(text)
        l = [
            {'value': options[0], 'time': 1.00, 'tokens': MAX_LEN},
            {'value': options[1], 'time': 1.00, 'tokens': MAX_LEN},
            {'value': options[2], 'time': 1.00, 'tokens': MAX_LEN}
        ]

        # save suggestions to df
        return jsonify(sentences=l)

    return jsonify({'error': 'Missing data!'})

# text selection notification
# not triggered by client when JS is set to REPORT_SELECTIONS_ON_SUBMIT = true
@app.route('/autocomplete-selection', methods=['POST'])
def handle_text_selection():
    data = request.json
    print("selection - got: " + json.dumps(data, indent = 4)) # TODO: log
    return jsonify({'status': 'ok'})

# text submission notification
@app.route('/submit', methods=['POST'])
def handle_submission():
    data = request.json
    print("submit - got: " + json.dumps(data, indent = 4)) # TODO: log

    # change accpeted option value (JS values are 01(first)/11(second)/21(third)/-1(skipped). should be --> 1/2/3 or -1)
    suggestions = data.get('suggestions', "")
    for s in suggestions:
        if s['accepted'] != -1:
            s['accepted'] = int(s['accepted'][0])+1

    status = save_session_to_file(data)
    return jsonify({'status': status})

# convert time (milisec) to datetime format
def convert_timestamp(timestamp):
    # try:
    #     # when timestamp is in seconds
    #     date = datetime.datetime.fromtimestamp(timestamp)
    # except (ValueError):
    # when timestamp is in miliseconds
    date = datetime.datetime.fromtimestamp(timestamp / 1000)
    return date

# save user data to SESSIONS_CSV (all sessions)
# in case reading or writing to file fails - save user session to temporary file
def save_session_to_file(data):
    # target file
    to_file = SESSIONS_CSV
    # target file in case of error
    temp_file = "./logs/fail/user_session_" + str(data.get('userid', "")) + "_" + datetime.datetime.today().strftime('%Y%m%d') + ".csv"
    if os.path.exists(SESSIONS_CSV):
        try:
            df_all_sessions = pd.read_csv(SESSIONS_CSV)
        except (IOError, Exception) as e:
            print(e)
            # error reading file - save to temp file
            df_all_sessions = pd.DataFrame(columns=["userid", "completionEnabled", "text", "timestamp", "suggestions"])
            to_file = temp_file
    else:
        df_all_sessions = pd.DataFrame(columns=["userid", "completionEnabled", "text", "timestamp", "suggestions"])

    # convert time (milisec) to datetime format
    data['timestamp'] = convert_timestamp(data['timestamp'])
    # df_all_sessions.loc[len(df_all_sessions.index)] = data
    df_all_sessions = df_all_sessions.append(data, ignore_index=True)

    # save to file
    try:
        df_all_sessions.to_csv(to_file, index=False)
    except (PermissionError, Exception) as e:
        print(e)
        # error - save to temp file only the current session
        new_data = pd.DataFrame(df_all_sessions[-1:].values, columns=df_all_sessions.columns)
        new_data.to_csv(temp_file, index=False)
        # return error to client
        return 'error'
    return 'ok'


if __name__ == "__main__":
    completer = Autocompleter("distilgpt2")

    # app.run(debug=True)
    app.run()
