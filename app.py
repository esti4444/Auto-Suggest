from autocompleter import Autocompleter
from flask import Flask, render_template, url_for, request, redirect, jsonify
#from flask_ngrok import run_with_ngrok
# if SQL:
# from flask_sqlalchemy import SQLAlchemy
# import pymysql
from datetime import datetime

first_timestamp = datetime.now()
num_of_tabs = 0
# max number of returned words in option
MAX_LEN = 3

# if SQL:
# # Connecting to DB
# connection = pymysql.connect(host='localhost',
#                              user='root',
#                              password='',
#                              db='auto_complete',
#                              charset='utf8mb4',
#                              cursorclass=pymysql.cursors.DictCursor)
# cursor = connection.cursor()  # execute query in python
#from flask_cors import CORS


app = Flask(__name__)
# run_with_ngrok(app)
# CORS(app)
# if SQL:
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:8080/auto_complete'
# db = SQLAlchemy(app)


# Insert and update user's data from interface to DB at the end of trial
@app.route('/', methods=['POST', 'GET'])
def index():
    print("------ Index- final save -------")
    if request.method == 'POST':
        title = request.form['title']
        real_text = request.form['real_text']

        try:

            # if SQL:
            # # Insert the final text at the end of the trial
            # query = "INSERT INTO `final_text`(id,text) VALUES (%s,%s)"
            # cursor.execute(query, (title, real_text))
            # connection.commit()
            #
            # # Calculate the time it took the user to choose an option
            # # Start time is based on initial options appearance
            # query = '''UPDATE `texts`
            #                     SET time_diff = (SELECT TimeDifference
            #                     FROM (SELECT (TIMEDIFF(A.maxtime,A.mintime)*1000) as TimeDifference ,A.text as Atext, A.id as Aid
            #                     FROM  (SELECT max(time) as maxtime, MIN(time)as mintime, id, text
            #                     FROM `texts`
            #                     GROUP BY text,id) as A) as B
            #                     WHERE texts.id= B.Aid and texts.text= B.Atext )
            #                     WHERE time_diff = 0'''
            #
            # cursor.execute(query)
            # connection.commit()
            #
            # # Calculate the total time of trial
            # final_timestamp = datetime.now()
            # total_time = ((final_timestamp - first_timestamp).total_seconds()) / 60.0
            #
            # # Update the total time in DB
            # query = '''UPDATE `final_text`
            #             SET total_time = %s
            #             WHERE final_text.id=%s'''
            #
            # cursor.execute(query, (total_time, title))
            # connection.commit()
            #
            # # Remove time redundancy
            # query = "DELETE x1 from texts x1 INNER JOIN texts x2 " \
            #         "WHERE x1.unique_id<x2.unique_id and x1.id=x2.id and x1.text=x2.text and x1.offer1=x2.offer1 " \
            #         "and x1.offer2=x2.offer2 and x1.offer3=x1.offer3"
            # cursor.execute(query)
            # connection.commit()
            #
            # # Calculate total amount of auto-complete offers given to the user
            # query = "UPDATE `final_text` " \
            #         "SET NUM_OF_OFFERS = (SELECT COUNT(id) FROM `texts` " \
            #         "WHERE texts.id = final_text.id  GROUP BY id)"
            # cursor.execute(query)
            # connection.commit()
            #
            # # Calculate total amount of selected auto-complete offers by the user
            # query = "UPDATE `final_text` " \
            #         "SET NUM_OF_SELECTED = (SELECT COUNT(id) FROM `texts` " \
            #         "WHERE texts.id = final_text.id and texts.selected_offer IS NOT NULL GROUP BY id)"
            # cursor.execute(query)
            # connection.commit()

            return 'Thanks For Participating'

        except:
            return 'There was an issue with adding your text'

    else:
        return render_template('index.html')
        #return render_template('Write With Transformer.html')

# Insert and update user's data from interface to DB during the trial

# "https://transformer.huggingface.co/autocomplete/distilgpt2/small"


#@app.route('/autocomplete/distilgpt2/small', methods=['POST'])
@app.route('/autocomplete', methods=['POST'])
def process():
    print("process request")
    data = request.json
    # title = data[0]['title']
    # text = data[0]['real_text']
    text = data['context']

    if text:
        completions, delta, options = completer.predict(text)
        l = [
            {'value': options[0], 'time': 1.00, 'tokens': MAX_LEN},
            {'value': options[1], 'time': 1.00, 'tokens': MAX_LEN},
            {'value': options[2], 'time': 1.00, 'tokens': MAX_LEN}
        ]
        return jsonify(sentences=l)
    return jsonify({'error': 'Missing data!'})


# Insert and update user's chosen offer from interface to DB during the trial
@app.route('/admin', methods=['POST', 'GET'])
def admin_index():
    print("------ Admin- step save -------")
    data = request.json

    title = data[0]['title']
    text = data[0]['real_text']
    selected_offer = data[0]['chosen']

    if text:

        # Actual dynamic options will be taken from predict function
        # Currently commented out until a prediction model is set in predict function
        offer1 = " is about"
        offer2 = " animals"
        offer3 = " and cats"

        # offer1 = predict(text, 4, token, model_test)
        # offer2 = predict(text, 5, token, model_test)
        # offer3 = predict(text, 6, token, model_test)

        # if SQL:
        # # Update the selected offer
        # query = "INSERT INTO `texts`(id,text,offer1,offer2,offer3,selected_offer) VALUES (%s,%s,%s,%s,%s,%s)"
        # cursor.execute(query, (title, text, offer1, offer2, offer3, selected_offer))
        # connection.commit()
        #
        # return jsonify({'text1': offer1, 'text2': offer2, 'text3': offer3})

    return jsonify({'error': 'Missing data!'})


if __name__ == "__main__":
    completer = Autocompleter("distilgpt2")

    # app.run(debug=True)
    app.run()
