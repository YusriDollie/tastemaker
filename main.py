import os
import sys
import argparse

from maps import FrozenMap

from classifiers.classify_songs import get_playlist_features, classify_playlist
from utils import spotutils, constants
from utils.data_processer import (
    scale_data,
    read_data,
    sanity_check,
    test_cluster_size,
    get_kmeans_clusters,
)
from utils.serialize import save_classifier, load_classifier, load_all_classifiers
from utils.spotutils import get_playlist_data
from utils.training_utils import (
    get_experiment_split,
    get_train_clusters,
    run_active_suite,
    run_clusters_suite,
    get_highest_benchmark,
)

import pandas


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--skipgen",
        action="store_true",
        help="skip taste data generation and read data from file",
    )
    parser.add_argument(
        "-c",
        "--skipclass",
        action="store_true",
        help="skip classifier data and read data from file",
    )
    args = parser.parse_args()

    user = input("Please enter Spotify Username:\n")
    print(f"Attempting to authenticate as {user}")

    sp, token = spotutils.login_to_spotify(user)
    print(token)

    if not args.skipgen:

        stuff = {}

        playlists = sp.user_playlists(user)
        while playlists:
            for i, playlist in enumerate(playlists["items"]):
                print(
                    "%4d %s %s"
                    % (i + 1 + playlists["offset"], playlist["uri"], playlist["name"])
                )
                stuff[i + 1] = str(playlist["uri"])
            if playlists["next"]:
                playlists = sp.next(playlists)
            else:
                playlists = None

        likeId = input("Enter ID of Playlist to Train Likes\n")
        dislikeID = input("Enter ID of Playlist to Train Dislikes\n")

        likedData = spotutils.get_playlist_data(sp, user, stuff[int(likeId)])[0]
        dislikedData = spotutils.get_playlist_data(sp, user, stuff[int(dislikeID)])[0]

        likesdf = spotutils.get_dataframe(sp, likedData, constants.LIKE)
        dislikesdf = spotutils.get_dataframe(sp, dislikedData, constants.DISLIKE)

        # print(likesdf)
        # print(dislikesdf)

        dataSet = pandas.concat([likesdf, dislikesdf], axis=0)

        scaledDataset = scale_data(dataSet)

        # dump to csv
        try:
            outdir = "./data"
            outname = "data.csv"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            fullname = os.path.join(outdir, outname)
            scaledDataset.to_csv(fullname, encoding="utf-8")
        except (IOError, FileNotFoundError):
            print("Failed to create data Output file")
            sys.exit(1)

        print("Data processing complete\n Outputted to data/data.csv")

    if not args.skipclass:

        print("Preparing Training data\n Reading in data from data/data.csv")
        baseData = FrozenMap(read_data("data/data.csv", True))

        print("Song data reading complete.")

        featureNames = tuple(next(iter(baseData.values()))["features"].keys())

        print("Running Data sanity checks")
        sanity_check(baseData)
        test_cluster_size(baseData, 10)

        print("Computing Clusters")

        clusteredData, songsByCluster = {}, {}
        clusteredData, songsByCluster = get_kmeans_clusters(
            baseData, constants.NUM_CLUSTERS
        )

        print("Running Cluster sanity checks")

        # sanity check
        if set(next(iter(baseData.values()))["features"].keys()) == set(
            next(iter(clusteredData.values()))["features"].keys()
        ):
            raise ValueError("Default features messed up.")

        print("Cluster sanity checks passed")

        (
            baseTrainingData,
            baseValidationData,
            clusteredTrainingData,
            clusteredValidationData,
        ) = get_experiment_split(baseData, clusteredData)
        trainingClusters = get_train_clusters(
            clusteredTrainingData, songsByCluster[constants.NUM_CLUSTERS]
        )

        print("----\nUnclustered\n----")
        activeUnclusteredResults = run_active_suite(
            baseData,
            baseTrainingData,
            baseValidationData,
            constants.SUPPORTED_ALGS,
            constants.AL_STRATS,
        )

        print("----\nClustered\n----")
        activeClusteredResults = run_active_suite(
            baseData,
            baseTrainingData,
            clusteredValidationData,
            constants.SUPPORTED_ALGS,
            constants.AL_STRATS,
        )

        print("\n----\nClustered w/ Cluster Sampling\n----")
        activeClusterSampledResults = run_clusters_suite(
            baseData, clusteredTrainingData, clusteredValidationData, trainingClusters
        )

        best_classifier = get_highest_benchmark(
            baseData,
            baseTrainingData,
            baseValidationData,
            clusteredTrainingData,
            clusteredValidationData,
        )

        save_classifier(best_classifier, None)

    filename = input(
        "Please enter a classifier filename or leave blank to load all, and press enter:\n"
    )

    if filename:
        CLASSIFIERS = [(load_classifier(filename), filename)]
    else:
        CLASSIFIERS = load_all_classifiers()

    print("\nClassifier(s) successfully loaded.")

    PLAYLIST = input("Please enter a Spotify playlist ID or URI to run predictions against, and press enter:\n")

    # You must specify a playlist
    # e.g. spotify:user:USERNAME:playlist:PLAYLIST_ID (Spotify URI)

    if not PLAYLIST:
        raise RuntimeError("Please specify a playlist.")

    # download Spotify playlist data
    playlistData, playlistName = get_playlist_data(
        sp, os.environ.get("SPOTIPY_CLIENT_ID"), PLAYLIST
    )

    print("\nPlaylist successfully downloaded.")

    playlistFeatureData, playlistDataDict = get_playlist_features(
        sp, playlistData, playlistName
    )

    print("Playlist successfully parsed.")

    for clf in CLASSIFIERS:
        classify_playlist(clf, playlistFeatureData, playlistDataDict)

