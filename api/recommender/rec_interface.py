class RecInterface:
  def __init__(self):
    pass

  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    raise NotImplementedError