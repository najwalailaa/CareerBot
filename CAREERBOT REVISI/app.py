from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Kunci rahasia 24 byte

@app.route('/', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        text = request.form.get('skill', type=str)
        # Membaca dataset 
        df = pd.read_csv('static/job_dataset.csv')
        # Menghapus baris yang kosong pada dataset
        df = df.dropna(axis='rows')
        df = df.fillna('unknown')
        # Membuat variabel baru untuk menyimpan variabel skill
        df_skill = df.Skill
        # Preprocessing data variabel skill
        def remove_brackets(data):
            data = str(data).lower()
            data = data.replace(',', '')
            data = data.replace('()', '')
            data = data.replace('-', '')
            return data

        df_skill = df_skill.apply(remove_brackets)
        # Melakukan transformasi TF-IDF pada variabel vector_tfidf
        vector_tfidf = TfidfVectorizer()
        # Menggabungkan data skill dengan text input dan indeks diatur ulang agar tetap berurutan
        df_skill_new = pd.concat([df_skill, pd.Series([text])], ignore_index=True)
        # Mengubah teks menjadi vektor yang sesuai dengan model TF-IDF dan menghitung frekuensi term
        result_tfidf = vector_tfidf.fit_transform(df_skill_new)
        # Mengubah hasil dari TF-IDF yang berbentuk matrix menjadi array numpy
        result_arr = result_tfidf.toarray()
        # Menyimpan model dan hasil transformasi TF-IDF ke file
        pickle.dump(vector_tfidf, open('static/model_tfidf.pkl', 'wb'))
        pickle.dump(result_arr, open('static/result_tfidf.pkl', 'wb'))
        # Inisialisasi untuk hasil
        result = {}
        # Menghitung kesamaan kosinus dengan mengabaikan vektor terakhir
        for id, vector in enumerate(result_arr[:-1]):
            cosine_val = cosine_similarity([result_arr[id]], [result_arr[-1]])
            result[id] = cosine_val[0][0]
        # Mengurutkan hasil berdasarkan nilai kesamaan kosinus
        result_desc = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        # Mengambil 5 urutan teratas yang memiliki cosine similarity > 0
        top_result_desc = {job_id: cosine_val for job_id, cosine_val in result_desc.items() if cosine_val > 0}
        top_result_desc = dict(list(top_result_desc.items())[:5])
        
        # Mengambil nama pekerjaan yang berkaitan
        top_jobs = [(df.iloc[job_id]['Pekerjaan'], cosine_val) for job_id, cosine_val in top_result_desc.items()]

        # Simpan hasil ke dalam sesi
        session['top_jobs'] = top_jobs

        return redirect(url_for('add'))

    # Mengambil hasil dari sesi, jika ada
    top_jobs = session.pop('top_jobs', [])

    return render_template('index.html', top_jobs=top_jobs)

if __name__ == '__main__':
    app.run(debug=True)
