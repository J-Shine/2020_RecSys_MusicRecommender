# Playlist Continuation

HAI Playlist Continuation Task

## 🔨 How to run  
Download the `song_meta.json` to `api/`  
After that, use these commands
```bash
$> pip install -r requirements.txt
$> npm install
$> npm run build
$> npm run start-api
```
Then, the server will open in `localhost:5000`  

## 🔎 How to Implement Recommender
Implement `inference` in `api/recommender/test_recommender.py`
```python
class RecInterface:
  def __init__(self):
    pass

  def inference(self, user_playlist):
    """Recommend Playlist with given user playlist
    :return: (List) song_ids of the recommended playlist
    """
    raise NotImplementedError
```