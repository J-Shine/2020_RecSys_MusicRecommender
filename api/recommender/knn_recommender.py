from .rec_interface import RecInterface


import numpy as np
import pandas as pd

import scipy.sparse as spr
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter




class KNNRecommender(RecInterface):
  train = pd.read_json("train.json")
  song_meta = pd.read_json("song_meta.json")
  train_data = train[['id','songs']]
  n_train = len(train_data)
  n_songs = len(song_meta)
  total = []
  def rec(self, user_playlist):
    amplifier = 2

    top500 = self.total[0].argsort()[-100:][::-1]
    p = np.zeros((self.n_songs, 1))
    for top in top500:
      suv = self.total[0][top]
      for song in self.train.loc[top, 'songs']:
        p[song] += pow(suv, amplifier)

    cand_song_idx = p.reshape(-1).argsort()[-100:][::-1]
    songs_already = user_playlist

    cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:30]
    rec_song_idx = [i for i in cand_song_idx]
    return rec_song_idx

  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    test_data = [{'id': 90000000, 'songs': user_playlist}]
    test = pd.DataFrame(data=test_data)
    print(test)
    plylst = pd.concat([self.train_data, test], ignore_index=True)
    print(plylst.tail())
    all_songs = plylst['songs']
    song_counter = Counter([song for songs in all_songs for song in songs])
    song_dict = {x: song_counter[x] for x in song_counter}

    plylst_use = plylst[['songs', 'id']]
    plylst_use.loc[:, 'num_songs'] = plylst_use['songs'].map(len)
    plylst_use['song_count'] = plylst_use['songs'].map(
      lambda x: [1 / ((song_dict.get(song) - 1) ** (0.44) + 1) for song in x])
    plylst_train = plylst_use.iloc[:-1, :]
    plylst_test = plylst_use.iloc[-1:, :]

    row = np.repeat(range(self.n_train), plylst_train['num_songs'])
    col = [song for songs in plylst_train['songs'] for song in songs]
    dat = np.repeat(1, plylst_train['num_songs'].sum())
    train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_songs))

    row2 = np.repeat(range(1), plylst_test['num_songs'])
    col2 = [song for songs in plylst_test['songs'] for song in songs]
    dat2 = np.repeat(1, plylst_test['num_songs'].sum())
    test_songs_A = spr.csr_matrix((dat2, (row2, col2)), shape=(1, self.n_songs))

    similarity = cosine_similarity(test_songs_A, train_songs_A)

    song_cound_data = np.concatenate(plylst_train['song_count'])
    row = np.repeat(range(self.n_train), plylst_train['num_songs'])
    col = [song for songs in plylst_train['songs'] for song in songs]
    train_songs_freq = spr.csr_matrix((song_cound_data, (row, col)), shape=(self.n_train, self.n_songs))
    frequency = test_songs_A.dot(train_songs_freq.T)
    frequency_array = frequency.toarray()
    self.total = frequency_array * similarity

    answers = self.rec(plylst_test.index)

    return answers

