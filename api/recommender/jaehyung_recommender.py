MAX_PLAYLISTS = 153430

from .rec_interface import RecInterface
from .j_shine.utils.datareader import DataReader
from .j_shine.recommender.dot_product import dot_product_similarity, dot_product
from .j_shine.utils.post_processing import eurm_to_recommendation_list
import pandas as pd
import numpy as np

class JaehyungRecommender(RecInterface):
  def __init__(self):
    self.dr = DataReader(only_load=True)

  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    # Concat user_playlist and train set dataframe
    self.user_df = pd.DataFrame(index=range(0, len(user_playlist)),
                                columns=["pid", "tid"])
    # Create new Dataframe using input from the user
    i = 0
    for song in user_playlist:
      self.user_df.iloc[i][0] = np.int32(MAX_PLAYLISTS - 1)
      self.user_df.iloc[i][1] = np.int32(song)
      i = i + 1

    """
    print("self.dr.df")
    print(self.dr.df)
    print("self.user_df")
    print(self.user_df)
    """

    self.dr.all_df = self.dr.df.append(self.user_df, ignore_index=True)
    """
    print("self.dr.all_df")
    print(self.dr.all_df)
    """

    # calculate score
    urm = self.dr.get_urm(only_load=False)
    s = dot_product_similarity(urm.T, k=100)
    r = dot_product(urm, s, k=500)
    r = r.tocsr()

    # recommend
    eurm = r[MAX_PLAYLISTS-1]
    rec_list = eurm_to_recommendation_list(eurm, datareader=self.dr)
    return rec_list[0]