# Music Recommender

*These music recommenders are created by the team of the Hanyang AI Society.*<br>
*Each team member developed their own recommender and merged them to one to demonstrate.*<br><br>

Here, I demonstrate works of mine.<br>
It is referencing a part of [the paper](https://dl.acm.org/doi/10.1145/3267471.3267475) and [the codes](https://github.com/tmscarla/spotify-recsys-challenge) written by Creamy Fireflies on Recsys Challenge 2018: Automatic Music Playlist Continuation.<br>


# Demo

https://user-images.githubusercontent.com/61873510/128503165-c55ff756-c96b-4013-a8cc-944d9e5a8167.mp4

# Full Pipeline

<img width="670" alt="pipeline" src="https://user-images.githubusercontent.com/61873510/128503030-2b597b0c-ab07-47d7-a1c4-a36bfd9d63ac.png">

# Steps

## Preprocessing

<img width="670" alt="preprocessing" src="https://user-images.githubusercontent.com/61873510/128853863-2cf223c6-99a2-4bba-a799-db42a4e67447.png">

Convert train, validate and test files from json to csv and merge them to one.<br>

<img width="289" alt="urm" src="https://user-images.githubusercontent.com/61873510/128856673-85a713d1-3c1f-455b-be6b-8623f696d481.png">

Then pick playlists column and tracks column to make sparse binary matrix, that is, user rating matrix.<br>

## Processing
<img width="382" alt="그림설명_3_processing" src="https://user-images.githubusercontent.com/61873510/128977616-2ec48c02-7fcf-4a66-bd84-0298c0f8b7b8.png">

### Similarity Matrix
<img width="670" alt="similarity matrix" src="https://user-images.githubusercontent.com/61873510/128973094-7fd65f6f-2390-4ebb-a636-93c554c2808f.png">
Make similarity matrix(variable s) by dot producting track vectors itself.<br>

### Score Matrix
<img width="770" alt="그림설명_3_score" src="https://user-images.githubusercontent.com/61873510/128976562-57b072a5-5187-49e4-98d8-a7b3fdc84d40.png">

Dot product urm and similarity matrix to get score matrix.<br>
Then choose top tracks to make recommend list.<br><br>
