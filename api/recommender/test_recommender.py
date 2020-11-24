from .rec_interface import RecInterface

class TestRecommender(RecInterface):
  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    return user_playlist