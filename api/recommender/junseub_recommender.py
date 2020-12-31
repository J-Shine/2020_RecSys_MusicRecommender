from .rec_interface import RecInterface


import numpy as np
import pandas as pd

import scipy.sparse as spr
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter




class JunseubRecommender(RecInterface):

  def __init__(self):
    self.train = pd.read_json("train.json")
    self.song_meta = pd.read_json("song_meta.json")
    self.train = self.train[['id','songs']]
    self.n_train = len(self.train)
    self.n_songs = len(self.song_meta)
    all_songs = self.train['songs']
    self.song_counter = Counter([song for songs in all_songs for song in songs])
    self.song_dict = {x: self.song_counter[x] for x in self.song_counter}
    self.total = []
    self.train.loc[:, 'num_songs'] = self.train['songs'].map(len)
    self.train['song_count'] = self.train['songs'].map(
      lambda x: [1 / ((self.song_dict.get(song) - 1) ** (0.44) + 1) for song in x])
    row = np.repeat(range(self.n_train), self.train['num_songs'])
    col = [song for songs in self.train['songs'] for song in songs]
    dat = np.repeat(1, self.train['num_songs'].sum())
    self.train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_songs))

    song_count_data = np.concatenate(self.train['song_count'])
    row = np.repeat(range(self.n_train), self.train['num_songs'])
    col = [song for songs in self.train['songs'] for song in songs]
    self.train_songs_freq = spr.csr_matrix((song_count_data, (row, col)), shape=(self.n_train, self.n_songs))

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
    test = [{'id': 90000000, 'songs': user_playlist, 'num_songs': len(user_playlist)}]
    plylst_use = pd.DataFrame(data=test)
    plylst_use['song_count'] = plylst_use['songs'].map(
      lambda x: [1 / ((self.song_dict.get(song) - 1) ** (0.44) + 1) for song in x])

    row2 = np.repeat(range(1), plylst_use['num_songs'])
    col2 = [song for songs in plylst_use['songs'] for song in songs]
    dat2 = np.repeat(1, plylst_use['num_songs'].sum())
    test_songs_A = spr.csr_matrix((dat2, (row2, col2)), shape=(1, self.n_songs))

    similarity = cosine_similarity(test_songs_A, self.train_songs_A)


    frequency = test_songs_A.dot(self.train_songs_freq.T)
    frequency_array = frequency.toarray()
    self.total = frequency_array * similarity

    answers = self.rec(user_playlist)

    return answers

