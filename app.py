import flask
import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import os
import csv
from flask import Flask, request, make_response,render_template,redirect,url_for


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return flask.render_template('index.html')

def cleaning(df):
    documents_clean = []
    for d in df:
        document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
        document_test = re.sub(r'@\w+', '', document_test)
        document_test = document_test.lower()
        document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
        document_test = re.sub(r'[0-9]', '', document_test)
        document_test = re.sub(r'\s{2,}', ' ', document_test)
        documents_clean.append(document_test)
    return documents_clean

def bow(data):
    tokens = data['clean'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokens = tokens.apply(lambda sem: [stemmer.stem(w) for w in sem])
    tokens = tokens.apply(lambda x: " ".join([w for w in x]))
    data['clean'] = tokens
    return data['clean']


def get_similar_articles(q, df, vectorizer, data):
    new = pd.DataFrame()
    cos_sim = []
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = {}
    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    for k, v in sim_sorted:
        if v != 0.0:
            cos_sim.append(v)
            w = data.loc[data['row_number'] == k]
            new = pd.concat([new, w])
    return cos_sim, new


def vector(q, data1, data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data1)
    X = X.T.toarray()
    df = pd.DataFrame(X, index=vectorizer.get_feature_names())
    return get_similar_articles(q, df, vectorizer, data)

def final(cos_sim , sd):
    sd = sd.drop(['desc', 'row_number', 'clean'], axis=1)
    sd['cosine_similarities']=cos_sim
    return sd


@app.route('/data', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':


        try:
            data = pd.read_csv(request.files.get('csvfile'))

            g = request.form.get('column_name')
            query = request.form.get("query")
            data['desc'] = data[g]
            data['row_number'] = np.arange(len(data))
            documents_clean = cleaning(data['desc'])
            data['clean'] = documents_clean
            data['clean'] = bow(data)
            cos_sim, sd = vector(query, data['clean'], data)
            final_data = final(cos_sim, sd)
            final_data = final_data.reset_index()
            final_data = final_data.drop(['index'], axis=1)

            # resp = make_response(final_data.to_csv())
            # resp.headers["Content-Disposition"] = "attachment; filename=cos_sim.csv"
            # resp.headers["Content-Type"] = "text/csv"
            # return resp
            try:
                return render_template('result.html', query=query,
                                       tables=[final_data.to_html(classes='data', header="true")])


            except:
                no_result = "NO RESULTS FOUND MATCHING YOUR QUERY"

                return render_template('data.html', no_result=no_result)

        except:
                no_result = "check your column name and file once again"

                return render_template('data.html', no_result=no_result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
