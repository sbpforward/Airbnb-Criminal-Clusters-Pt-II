from flask import Flask, render_template, url_for
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    # Logic for reading our data
    data = 'placeholder for table'
    return render_template('index.html', data=data)

@app.route('/test')
def test():
    # Logic for reading our data
    df = pd.read_csv('static/data/data.csv')
    data = df.to_html()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run()

