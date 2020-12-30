# -*- coding: utf-8 -*-
from .rec_interface import RecInterface
from collections import Counter
import numpy as np
import scipy.sparse as spr
import pandas as pd
import copy

class SimpleKNNRecommender(RecInterface):
    song_meta = pd.read_json("song_meta.json")
    train = pd.read_json("train.json")
    train_plylist = train[['id', 'songs']]
    n_train = len(train)
    n_songs = len(song_meta)

    train_songs_A=[]
    test_songs_A = []

    def rec(self, test_id):
        p1 = copy.deepcopy(self.test_songs_A)
        pt = p1.T
        p = pt.toarray()
        songs_already = test_id

        simpls = self.train_songs_A.dot(p)
        simpls2 = np.zeros_like(simpls)

        inds = simpls.reshape(-1).argsort()[-100:][::-1]  # .reshape(-1) == .reshape(1, -1)처럼 1차원 배열 반환
        vals = simpls[inds]

        m = np.max(vals)
        if (m == 0):
            m += 0.01

        vals2 = ((vals - np.min(vals)) * (1 / m)) ** 2
        simpls2[inds] = vals

        train_songs_A_T = self.train_songs_A.T.tocsr()

        cand_song = train_songs_A_T[:, inds].dot(vals2)
        cand_song_idx = cand_song.reshape(-1).argsort()[-50:][::-1]

<<<<<<< HEAD
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][ :15]  # playlist에 원래 있던 song이 아닌 것들 15개
=======
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][ :30]  # playlist에 원래 있던 song이 아닌 것들 30개
>>>>>>> 7073362a148c822785cc2779385cb5360178a0e2
        rec_song_idx = [i for i in cand_song_idx]

        return rec_song_idx

    def inference(self, user_playlist):
        """Recommend Playlist with given user playlist (user_playlist를 song id list로 받는다.)
        :return: (List) song_ids of the recommended playlist (추천할 song id list를 return하도록 구현)
        """
        test_plylist = [{'id':120000, 'songs':user_playlist}] #train의 playlist id가 115071번까지 있어서 120000
        test = pd.DataFrame(data=test_plylist)
        plylst = pd.concat([self.train_plylist, test], ignore_index=True) # train + test

        plylst_song = plylst['songs']  #playlist에 있는 노래들
        song_counter = Counter([sg for sgs in plylst_song for sg in sgs])  # type: collections.Counter
        song_dict = {x: song_counter[x] for x in song_counter}  # type: dict

        plylst_use = plylst[['id', 'songs']]
        plylst_use.loc[:, 'num_songs'] = plylst_use['songs'].map(len)  # playlist의 song개수 추가

        plylst_train = plylst_use.iloc[:-1, :]
        plylst_test = plylst_use.iloc[-1:, :]

        row = np.repeat(range(self.n_train), plylst_train['num_songs'])
        col = [song for songs in plylst_train['songs'] for song in songs]
        dat = np.repeat(1, plylst_train['num_songs'].sum())
        self.train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_songs))
        # -> train playlist와 song의 co-occurence matrix

        test_row = np.repeat(range(1), plylst_test['num_songs'])
        test_col = [song for songs in plylst_test['songs'] for song in songs]
        test_dat = np.repeat(1, plylst_test['num_songs'].sum())
        self.test_songs_A = spr.csr_matrix((test_dat, (test_row, test_col)), shape=(1, self.n_songs))
        # -> test playlist와 song의 co-occurence matrix

        answers = self.rec(plylst_test.index)
        return answers
