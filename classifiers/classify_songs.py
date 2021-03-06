"""
classify songs given a playlist / final function
"""

from maps import FrozenMap  # pylint: disable=import-error

from utils import spotutils, data_processer


def get_playlist_features(SPOTIPY_OBJECT, playlist_data, playlist_name):
    # featurize playlist song data
    playlist_df = spotutils.get_dataframe(SPOTIPY_OBJECT, playlist_data, -1)

    scaled_playlist_df = data_processer.scale_data(playlist_df)
    scaled_playlist_df.to_csv("data/{}.csv".format(playlist_name), encoding="utf-8")

    playlist_data_dict = FrozenMap(
        data_processer.read_data("data/{}.csv".format(playlist_name), False)
    )
    return data_processer.get_features_and_id(playlist_data_dict), playlist_data_dict


def classify_playlist(classifier, playlist_feature_data, playlist_data_dict):
    """
    take in playlist, output songs they'd like using classifier
    """

    clf, clf_name = classifier

    playlist_features, playlist_song_ids = playlist_feature_data
    # run classifier on playlist songs
    results = clf.predict_all(playlist_features)
    liked_songs = [playlist_song_ids[i] for i in range(len(results)) if results[i] == 1]

    # get songs they'd like based on song ID
    if not liked_songs:
        print(
            "The classifier "
            + clf_name
            + " thinks you wouldn't like any songs in \
            the given playlist."
        )
        return

    print(
        "The classifier "
        + clf_name
        + " thinks you'd like the following from the given playlist:\n"
    )

    for song in liked_songs:
        print(playlist_data_dict[song]["metadata"]["track_name"])

    numLiked = len(liked_songs)
    totalSongs = len(playlist_song_ids)
    matchRate = numLiked / totalSongs * 100

    print(
        "The classifier "
        + clf_name
        + " thinks you'd dislike the following from the given playlist:\n"
    )
    for song in playlist_song_ids:
        if song not in liked_songs:
            print(playlist_data_dict[song]["metadata"]["track_name"])

    # spotify:playlist:37i9dQZF1DWXJfnUiYjUKT

    # spotify:playlist:37i9dQZF1DXcRXFNfZr7Tp

    print(f"Thats a taste match of {matchRate}%")
    print()
    return
