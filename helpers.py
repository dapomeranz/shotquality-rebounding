import pandas as pd
import numpy as np


def clean_df(locs_path, pbp_path):
    locations = pd.read_csv(locs_path)
    play_by_play = pd.read_csv(pbp_path)
    # Feels like its cheating the problem at hand to predict based on teams or conferences
    play_by_play = play_by_play.drop(
        ["team", "opponent", "conference", "opp_conference"], axis=1
    )
    # Combine CSV data into one large table
    merged_df = pd.merge(play_by_play, locations, on="id", how="inner")
    # Convert coordinates to only reference one direction of the court
    # For now lets assume that if the shooter is on the right side of the court, the right basket is the target
    # Ignore half court shots+ for now
    # If the shooter has x greater than 47, mirror all points
    shooter_locations = locations[(locations["annotation_code"] == "s")]
    merged_df_2 = pd.merge(
        merged_df, shooter_locations, on="id", how="inner", suffixes=("", "_shooter")
    )
    # if merged_df_2['court_x_shooter'] > 47, then mirror court_x to be 94 - court_x
    merged_df_2["court_x"] = merged_df_2.apply(
        lambda x: 94 - x["court_x"] if x["court_x_shooter"] > 47 else x["court_x"],
        axis=1,
    )
    df = merged_df_2.drop(
        ["court_x_shooter", "court_y_shooter", "annotation_code_shooter"], axis=1
    )

    df["dtb"] = np.linalg.norm(df[["court_x", "court_y"]].values - [4, 25], axis=1)
    # Add rankings
    df["dtb_rank"] = df.groupby(["id"])["dtb"].rank(method="first").astype(int)
    df["team"] = np.where(df["annotation_code"].str.contains("d"), "Defense", "Offense")
    df["dtb_team_rank"] = (
        df.groupby(["id", "team"])["dtb"].rank(method="first").astype(int)
    )

    # Pivot the dataframe for each play
    pivot_df = df.pivot_table(
        index=["id", "is_oreb"],
        columns=["annotation_code"],
        values=["court_x", "court_y", "dtb"],
    )
    pivot_df.columns = ["{}_{}".format(col[0], col[1]) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    pivot_df_2 = df.pivot_table(
        index=["id", "is_oreb"], columns=["dtb_rank"], values=["dtb"]
    )
    pivot_df_2.columns = ["{}_{}".format(col[0], col[1]) for col in pivot_df_2.columns]
    pivot_df_2 = pivot_df_2.reset_index()

    pivot_df_3 = df.pivot_table(
        index=["id", "is_oreb"], columns=["dtb_team_rank", "team"], values=["dtb"]
    )
    pivot_df_3.columns = [
        "{}_{}_{}".format(col[0], col[1], col[2]) for col in pivot_df_3.columns
    ]
    pivot_df_3 = pivot_df_3.reset_index()

    pivot_df = pd.merge(pivot_df, pivot_df_2, on=["id", "is_oreb"], how="inner")
    pivot_df = pd.merge(pivot_df, pivot_df_3, on=["id", "is_oreb"], how="inner")

    # Add some aggregate columns
    pivot_df["total_offense_dtb"] = pivot_df[
        ["dtb_t1", "dtb_t2", "dtb_t3", "dtb_t4", "dtb_s"]
    ].sum(axis=1)
    pivot_df["total_defense_dtb"] = pivot_df[
        ["dtb_d1", "dtb_d2", "dtb_d3", "dtb_d4", "dtb_d5"]
    ].sum(axis=1)
    pivot_df["top2_offense_dtb"] = pivot_df[["dtb_1_Offense", "dtb_2_Offense"]].sum(
        axis=1
    )
    pivot_df["top3_offense_dtb"] = pivot_df[
        ["dtb_1_Offense", "dtb_2_Offense", "dtb_3_Offense"]
    ].sum(axis=1)
    pivot_df["top4_offense_dtb"] = pivot_df[
        ["dtb_1_Offense", "dtb_2_Offense", "dtb_3_Offense", "dtb_4_Offense"]
    ].sum(axis=1)
    pivot_df["top2_offense_dtb"] = pivot_df[["dtb_1_Offense", "dtb_2_Offense"]].sum(
        axis=1
    )
    pivot_df["top3_offense_dtb"] = pivot_df[
        ["dtb_1_Offense", "dtb_2_Offense", "dtb_3_Offense"]
    ].sum(axis=1)
    pivot_df["top4_offense_dtb"] = pivot_df[
        ["dtb_1_Offense", "dtb_2_Offense", "dtb_3_Offense", "dtb_4_Offense"]
    ].sum(axis=1)
    pivot_df["top1_any_dtb"] = pivot_df[["dtb_1"]].sum(axis=1)
    pivot_df["top2_any_dtb"] = pivot_df[["dtb_1", "dtb_2"]].sum(axis=1)
    pivot_df["top3_any_dtb"] = pivot_df[["dtb_1", "dtb_2", "dtb_3"]].sum(axis=1)
    pivot_df["top4_any_dtb"] = pivot_df[["dtb_1", "dtb_2", "dtb_3", "dtb_4"]].sum(
        axis=1
    )
    pivot_df["top5_any_dtb"] = pivot_df[
        ["dtb_1", "dtb_2", "dtb_3", "dtb_4", "dtb_5"]
    ].sum(axis=1)
    pivot_df["top6_any_dtb"] = pivot_df[
        ["dtb_1", "dtb_2", "dtb_3", "dtb_4", "dtb_5", "dtb_6"]
    ].sum(axis=1)
    pivot_df["top7_any_dtb"] = pivot_df[
        ["dtb_1", "dtb_2", "dtb_3", "dtb_4", "dtb_5", "dtb_6", "dtb_7"]
    ].sum(axis=1)
    pivot_df["top8_any_dtb"] = pivot_df[
        ["dtb_1", "dtb_2", "dtb_3", "dtb_4", "dtb_5", "dtb_6", "dtb_7", "dtb_8"]
    ].sum(axis=1)
    pivot_df["top9_any_dtb"] = pivot_df[
        [
            "dtb_1",
            "dtb_2",
            "dtb_3",
            "dtb_4",
            "dtb_5",
            "dtb_6",
            "dtb_7",
            "dtb_8",
            "dtb_9",
        ]
    ].sum(axis=1)
    count_less_than_5 = lambda row: sum(row < 5)
    pivot_df["count_less_than_5"] = (
        pivot_df[
            [
                "dtb_1",
                "dtb_2",
                "dtb_3",
                "dtb_4",
                "dtb_5",
                "dtb_6",
                "dtb_7",
                "dtb_8",
                "dtb_9",
                "dtb_10",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_5, axis=1)
    )
    pivot_df["count_less_than_5_offense"] = (
        pivot_df[
            [
                "dtb_1_Offense",
                "dtb_2_Offense",
                "dtb_3_Offense",
                "dtb_4_Offense",
                "dtb_5_Offense",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_5, axis=1)
    )
    pivot_df["count_less_than_5_defense"] = (
        pivot_df[
            [
                "dtb_1_Defense",
                "dtb_2_Defense",
                "dtb_3_Defense",
                "dtb_4_Defense",
                "dtb_5_Defense",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_5, axis=1)
    )
    count_less_than_10 = lambda row: sum(row < 5)
    pivot_df["count_less_than_10"] = (
        pivot_df[
            [
                "dtb_1",
                "dtb_2",
                "dtb_3",
                "dtb_4",
                "dtb_5",
                "dtb_6",
                "dtb_7",
                "dtb_8",
                "dtb_9",
                "dtb_10",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_10, axis=1)
    )
    pivot_df["count_less_than_10_offense"] = (
        pivot_df[
            [
                "dtb_1_Offense",
                "dtb_2_Offense",
                "dtb_3_Offense",
                "dtb_4_Offense",
                "dtb_5_Offense",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_10, axis=1)
    )
    pivot_df["count_less_than_10_defense"] = (
        pivot_df[
            [
                "dtb_1_Defense",
                "dtb_2_Defense",
                "dtb_3_Defense",
                "dtb_4_Defense",
                "dtb_5_Defense",
            ]
        ]
        .iloc[:, 1:]
        .apply(count_less_than_10, axis=1)
    )
