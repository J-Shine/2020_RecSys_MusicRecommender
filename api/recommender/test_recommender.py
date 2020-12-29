from .rec_interface import RecInterface

class TestRecommender(RecInterface):
  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist

    userplaylist를 song id에 list로 받는다. 추천할 song id list를 뱉도록 구현하면 된다.
    """
    # print(user_playlist)
    return user_playlist