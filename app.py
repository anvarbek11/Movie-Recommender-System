import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



_data = None
_similarity = None
_suggestions = None

def create_similarity():
    """Optimized similarity matrix creation with memory efficiency"""
    global _data, _similarity, _suggestions

    # Load data with optimized data types
    _data = pd.read_csv('main_data.csv',
                        dtype={'comb': 'string',
                               'movie_title': 'category'})  # Reduce memory usage

    # Create count matrix with limited features
    cv = CountVectorizer(binary=True, max_features=8000)  # Reduced from default
    count_matrix = cv.fit_transform(_data['comb'])

    # Use sparse matrix for memory efficiency
    _similarity = cosine_similarity(count_matrix, dense_output=False)

    # Precompute suggestions
    _suggestions = _data['movie_title'].str.capitalize().tolist()



def rcmd(m):
    """Optimized recommendation function with numpy improvements"""
    global _data, _similarity

    m = m.lower()

    # Lazy load data if not loaded
    if _data is None or _similarity is None:
        create_similarity()

    if m not in _data['movie_title'].unique():
        return 'Sorry! Try another movie name'

    # Find index using categorical optimization
    idx = _data[_data['movie_title'] == m].index[0]

    # Sparse matrix row access
    row = _similarity[idx].toarray().flatten()

    # Optimized sorting with numpy
    top_indices = np.argpartition(row, -11)[-11:]  # Get top 11
    top_indices = top_indices[np.argsort(row[top_indices])][::-1][1:]  # Sort and exclude self

    return _data.iloc[top_indices]['movie_title'].tolist()


# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list


# to get suggestions of movies
def get_suggestions():
    """Get cached suggestions to avoid repeated CSV loading"""
    global _suggestions
    if _suggestions is None:
        create_similarity()
    return _suggestions


# Flask API

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    return rc if isinstance(rc, str) else "---".join(rc)


@app.route("/recommend", methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in
                    range(len(cast_places))}


    # passing all the data to the html file
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
                           vote_count=vote_count, release_date=release_date, runtime=runtime, status=status,
                           genres=genres,
                           movie_cards=movie_cards, casts=casts, cast_details=cast_details)


if __name__ == '__main__':
    app.run(debug=True)