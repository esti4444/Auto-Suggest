from autocompleter import Autocompleter
from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import os
import datetime

# max number of returned words in option
MAX_LEN = 3
SESSIONS_CSV = "./all_sessions.csv"

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
    status = save_session_to_file(data)
    return jsonify({'status': status})

def convert_timestamp(timestamp):
    # try:
    #     # when timestamp is in seconds
    #     date = datetime.datetime.fromtimestamp(timestamp)
    # except (ValueError):
    # when timestamp is in miliseconds
    date = datetime.datetime.fromtimestamp(timestamp / 1000)
    return date

def save_session_to_file(data):
    if os.path.exists(SESSIONS_CSV):
        df_all_sessions = pd.read_csv(SESSIONS_CSV)
    else:
        df_all_sessions = pd.DataFrame(columns=["userid", "completionEnabled", "text", "timestamp", "suggestions"])

    data['timestamp'] = convert_timestamp(data['timestamp'])
    df_all_sessions.loc[len(df_all_sessions.index)] = data
    # df_all_sessions['timestamp'].apply(convert_timestamp)
    try:
        df_all_sessions.to_csv(SESSIONS_CSV, index=False)
    except (PermissionError):
        print("CSV save failed --- File permission error")
        return 'error'
    return 'ok'

    # df_user_session = pd.DataFrame(
    #     columns=["userid", "timestamp", "input_text", "option1", "option2", "option3", "accepted_option"])


if __name__ == "__main__":
    completer = Autocompleter("distilgpt2")


    # app.run(debug=True)
    app.run()
