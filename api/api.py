from flask import Flask, request
import json
from difflib import SequenceMatcher
from youtubesearchpython import SearchVideos

from recommender.test_recommender import TestRecommender
from recommender.haeun_recommender import HaeunRecommender
from recommender.jaehyung_recommender import JaehyungRecommender
from recommender.junseub_recommender import JunseubRecommender

app = Flask(__name__, static_folder="../build", static_url_path='/')
recommenders = [HaeunRecommender(), JaehyungRecommender(), JunseubRecommender()]

with open('song_meta.json') as json_file:
  songs = json.load(json_file)

songs_parsed = []

for item in songs:
  artist_string = '['
  for idx, artist in enumerate(item['artist_name_basket']):
    if idx==0:
      artist_string += artist
    else:
      artist_string += (', ' +artist)
  artist_string += '] '

  songs_parsed.append({
    'label': artist_string + item['song_name'],
    'value': item['id']
  })

@app.route('/')
def index():
  return app.send_static_file('index.html')

@app.route('/api/search', methods=['GET'])
def search():
  title = request.args.get('title')

  items = []
  for item in songs_parsed:
    if all(x in item['label'] for x in title.split()):
      items.append(item)
  
  items = items[:20]

  return {"result" : items}

@app.route('/api/recommendation', methods=['POST'])
def recommendation():
  data = request.json
  user_playlist = [item['value'] for item in data['playlist']]
  engine_id = data['engineId']
  recommendation = recommenders[engine_id].inference(user_playlist)

  parsed_recommendation = []
  for item in songs_parsed:
    if item['value'] in recommendation:
      parsed_recommendation.append(item)

  output = []

  for item in parsed_recommendation:
    search = SearchVideos(item['label'], mode = 'dict', language = 'ko-KR', region = "KR").result()
    search = search["search_result"][0]["id"]
    output.append({
      'label': item['label'],
      'value': item['value'],
      'videoId': search
    })

  return {"result" : output}

if __name__ == "__main__":              
  app.run(host="0.0.0.0", port=5000)