from flask import Flask, render_template, url_for
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    # Logic for reading data
    df = pd.read_json('static/data/flask_df.json')
    # table_columns = [
    #     'Likelihood',
    #     'Listing URL',
    #     'Listing in Denver',
    #     'Host ID',
    #     'Host in Denver',
    #     'Host URL',
    #     'Requires License',
    #     'Current License', 
    #     'Minimum Nights',
    #     'Maximum Nights'
    #     ]
    # df = df[table_columns]
    data = df[:40].to_html(index=False, classes='table table-striped table-dark')
    return render_template('index.html', data=data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/next-steps')
def test():
    return '<h1>Next Steps</h1>'

if __name__ == '__main__':
    app.run(debug=True)

