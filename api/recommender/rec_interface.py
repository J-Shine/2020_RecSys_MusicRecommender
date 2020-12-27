MAX_PLAYLISTS = 153430

from ...src.utils.datareader import DataReader
from ...src.recommender.dot_product import dot_product_similarity, dot_product
from ...src.utils.post_processing import eurm_to_recommendation_list
import pandas as pd

class RecInterface:
  def __init__(self):
    self.dr = DataReader(only_load=True)

  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    # Concat user_playlist and train set dataframe
    self.user_df = pd.DataFrame(index=range(0, len(user_playlist)),
                                columns=["playlist_id", "song_id"])
    i = 0
    for song in user_playlist:
      self.user_df.iloc[i][0] = MAX_PLAYLISTS
      self.user_df.iloc[i][1] = song
      i = i + 1

    self.dr.all_df = pd.concat([self.dr.df, self.user_df], axis=0, join='outer')

    # calculate score
    urm = self.dr.get_urm(only_load=False)
    s = dot_product_similarity(urm.T, k=100)
    r = dot_product(urm, s, k=500)
    r = r.tocsr()

    # recommend
    eurm = r[MAX_PLAYLISTS-1]
    rec_list = eurm_to_recommendation_list(eurm, datareader=self.dr)
    return rec_list[0]