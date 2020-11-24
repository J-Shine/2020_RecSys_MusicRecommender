import React, { useState, useEffect } from 'react';
import ReactJkMusicPlayer from 'react-jinke-music-player';
import 'react-jinke-music-player/assets/index.css';
import AsyncSelect from 'react-select/async';
import makeAnimated from 'react-select/animated';
import YouTube from 'react-youtube'
import './MusicPlayer.css'
import List from 'react-list-select'

const animatedComponent = makeAnimated()

function MusicPlayer() {
  const [selectedPlaylist, setSelectedPlaylist] = useState([]);
  const [playlist, setPlaylist] = useState([]);
  const [videoId, setVideoId] = useState("");

  const loadOptions = async (inputText, callback) => {
    const response = await fetch(`/api/search?title=${inputText}`)
    const json = await response.json()
    console.log(json)
    callback(json.result)
  }

  const onClick = async () => {
    setPlaylist([{'label':'Inferencing...', 'value':0}])
    const response = await fetch('/api/recommendation', {
      method: 'POST',
      cache: 'no-cache',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(selectedPlaylist)
    }).then(response => response.json());
    setPlaylist(response.result);
  }

  const onChange = (selected) => {
    setVideoId(playlist[selected].videoId)
  }

  return (
    <>
      <h1>HAI Music Recommendation</h1>
      <div style={{display: 'flex'}}>
        <div style={{flex:100}}>
          <AsyncSelect
            isMulti
            components={animatedComponent}
            value={selectedPlaylist}
            onChange={setSelectedPlaylist}
            placeholder={'type your playlist...'}
            loadOptions={loadOptions}
          />
        </div>
        <div style={{flex:1}}>
          <button onClick={onClick}>
            Submit
          </button>
        </div>
      </div>
      <YouTube videoId={videoId} containerClassName={"youtubeContainer"} />
      <h3>Recommended Playlists</h3>
      <List
        items={playlist.map(item => item.label)}
        onChange={onChange}
      />
    </>
  )
}

export default MusicPlayer;