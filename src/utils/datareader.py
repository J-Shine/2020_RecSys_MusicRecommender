import os
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp


NUM_TEST_SONGS = 197648
NUM_TRAIN_SONGS = 5285871
NUM_VAL_PLAYLISTS = 23015
NUM_VAL_SONGS = 421199
NUM_TRAIN_PLAYLISTS = 115071
MAX_PLAYLISTS = 153430
NUM_TRACKS = 707989

class DataReader:
    def __init__(self, train_fname = "../data/data_json/train.json",
                 val_fname = "../data/data_json/val.json",
                 test_fname = "../data/data_json/test.json",
                 only_load = False):
        # only load urm if only_load mode is True
        if only_load:
            train_fname_csv = "../data/data_csv/train.csv"
            val_fname_csv = "../data/data_csv/val.csv"
            val_pid_fname_csv = "../data/data_csv/val_pid.csv"
            test_fname_csv = "../data/data_csv/test.csv"
            all_fname_csv = "../data/data_csv/all.csv"

            if os.path.isfile(train_fname_csv) and os.path.isfile(val_fname_csv)\
                    and os.path.isfile(val_pid_fname_csv) and os.path.isfile(test_fname_csv):
                self.train_df = self.load_csv(train_fname_csv)
                self.val_df = self.load_csv(val_fname_csv)
                self.val_pid_df = self.load_val_pid_csv(val_pid_fname_csv)
                self.test_df = self.load_csv(test_fname_csv)
                self.df = self.load_csv(all_fname_csv)
            else:
                print("Error: File doesn't exist")



        """
        val.json에서 playlist_id만 추출하여 dataframe 만들고 csv로 저장
        """
        if not only_load:
            # read validate set
            val_list = self.load_json(val_fname)
            self.val_df = pd.DataFrame(index=range(0, NUM_VAL_SONGS),
                                         columns=["playlist_id","song_id"])
            i = 0
            for dic in val_list:
                for (k, v) in dic.items():
                    if(k == "id"):
                        playlist_id = v
                    if(k == "songs"):
                        for song_id in v:
                            self.val_df.iloc[i][0] = playlist_id
                            self.val_df.iloc[i][1] = song_id
                            i = i + 1

            self.val_df.rename(columns={'playlist_id':'pid',
                                          'song_id':'tid'},inplace=True)
            # csv로 저장
            self.val_df.to_csv("../data/data_csv/val.csv")
            print("val.json -> val.csv done")

            # read validate set
            val_pid_list = self.load_json(val_fname)
            self.val_pid_df = pd.DataFrame(index=range(0, NUM_VAL_PLAYLISTS),
                                       columns=["playlist_id"])
            i = 0
            for dic in val_pid_list:
                for (k, v) in dic.items():
                    if (k == "id"):
                        self.val_pid_df.iloc[i][0] = v
                        i = i + 1

            self.val_pid_df.rename(columns={'playlist_id': 'pid'}, inplace=True)

            # csv로 저장
            self.val_pid_df.to_csv("../data/data_csv/val_pid.csv")
            print("val.json -> val_pid.csv done")

            """
            train.json에서 playlist_id와 song_id만 추출하여 dataframe 만들고 csv로 저장
            """
            train_list = self.load_json(train_fname)
            self.train_df = pd.DataFrame(index=range(0, NUM_TRAIN_SONGS),
                                         columns=["playlist_id", "song_id"])
            i = 0
            for dic in train_list:
                for (k, v) in dic.items():
                    if(k == "id"):
                        playlist_id = v
                    if(k == "songs"):
                        for song_id in v:
                            self.train_df.iloc[i][0] = playlist_id
                            self.train_df.iloc[i][1] = song_id
                            i = i + 1

            self.train_df.rename(columns={'playlist_id':'pid',
                                          'song_id':'tid'},inplace=True)
            # csv로 저장
            self.train_df.to_csv("../data/data_csv/train.csv")
            print("train.json -> train.csv done")

            """
            test.json에서 playlist_id와 song_id만 추출하여 dataframe 만들고 csv로 저장
            """
            test_list = self.load_json(test_fname)
            self.test_df = pd.DataFrame(index=range(0, NUM_TEST_SONGS),
                                         columns=["playlist_id", "song_id"])
            i = 0
            for dic in test_list:
                for (k, v) in dic.items():
                    if(k == "id"):
                        playlist_id = v
                    if(k == "songs"):
                        for song_id in v:
                            self.test_df.iloc[i][0] = playlist_id
                            self.test_df.iloc[i][1] = song_id
                            i = i + 1

            self.test_df.rename(columns={'playlist_id':'pid',
                                          'song_id':'tid'},inplace=True)
            # csv로 저장
            self.test_df.to_csv("../data/data_csv/test.csv")
            print("test.json -> test.csv done")
            self.df = pd.concat([self.train_df, self.val_df, self.test_df], axis=0, join='outer')
            self.df.to_csv("../data/data_csv/all.csv")
            print("all.csv done")

    def get_urm(self, only_load = False):
        # only load urm if only_load mode is True
        if only_load:
            file_full_path = '../matrices/ptm.npz'
            if os.path.isfile(file_full_path):
                urm = self.__load_matrix(file_full_path)
                return urm

        # concat train_df and val_df
        # union of the train, val and test
        # concat은 그냥 init할 때 하기로 함(오래걸리므로)
        # self.df = pd.concat([self.train_df, self.val_df, self.test_df], axis=0, join='outer')

        # collect data to build urm
        playlists = self.all_df['pid'].values
        tracks = self.all_df['tid'].values
        """
        print("playlists")
        print(playlists)
        print("tracks")
        print(tracks)
        """
        assert (playlists.size == tracks.size)
        n_playlists = MAX_PLAYLISTS
        n_tracks = NUM_TRACKS
        n_interactions = tracks.size
        """
        print("n_playlists")
        print(n_playlists)
        print("n_tracks")
        print(n_tracks)
        print("n_interactions")
        print(n_interactions)
        """
        # building the user-rating matrix(playlist-track matrix)
        urm = sp.csr_matrix((np.ones(n_interactions), (playlists,tracks)),
                            shape=(n_playlists, n_tracks), dtype=np.int32)
        # save urm
        self.__save_matrix('ptm',urm)

        return urm


    def load_csv(self, fname):
        loaded = pd.read_csv(filepath_or_buffer=fname,
                         sep=',', header=0,
                         usecols=['pid', 'tid'],
                         dtype={'pid': np.int32, 'tid': np.int32})
        return loaded

    def load_val_pid_csv(self, val_fname):
        loaded = pd.read_csv(filepath_or_buffer=val_fname,
                         sep=',', header=0,
                         usecols=['pid'],
                         dtype={'pid': np.int32})
        return loaded

    def load_json(self, fname):
        with open(fname, encoding="utf-8") as f:
            json_obj = json.load(f)

        return json_obj

    def __save_matrix(self, name, sparse_matrix):
        if not os.path.exists('../matrices/'):
            os.makedirs('../matrices/')
            print("saving matrix...")
        sp.save_npz('../matrices/' + name+ '.npz', sparse_matrix)

    def __load_matrix(self, file_path):
        npz =  sp.load_npz(file_path).tocsr()
        return npz

    def get_val_playlists(self):
        """
        :param name /.../num_tracks:  if true it will be in the numpy array returned
        :return:            numpy ndarray of shape (10k, 2/3/4/5 )
                                                    (playlists id, features[name...numtracks]
        """
        df = pd.read_csv(filepath_or_buffer='../data/data_csv/val_pid.csv',
                         sep=',', header=0,
                         usecols=['pid'],
                         dtype={'pid': np.int32})

        # building info
        i = 0
        p_info = [df['pid'].values]
        order = str(i) + '-pid'
        i += 1

        p_info = np.array(p_info).T
        return p_info

    def get_val_pids(self):
        return self.get_val_playlists().transpose()[0]