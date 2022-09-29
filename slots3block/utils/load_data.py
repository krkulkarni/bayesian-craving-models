import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import glob
import copy
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_longform(path_to_data, n_trials=35, remove_below_chance=False):

    # Load all csvs into li
    all_jsons = glob.glob(path_to_data + "*.json")

    df_summary = pd.DataFrame()
    longform = pd.DataFrame()

    for i, subject_file in enumerate(all_jsons):
        with open(subject_file) as f:
            data = json.load(f)

        subj_df_summary = pd.DataFrame.from_dict(
            {
                "Run": [data["instructions"]["run"][-1]],
                "PID": [data["instructions"]["participant_id"]],
                "Block 1": [data["instructions"]["block_order"][0]],
                "Block 2": [data["instructions"]["block_order"][1]],
                "Block 3": [data["instructions"]["block_order"][2]],
                "Reversal Timings": [data["instructions"]["reversal_timings"]],
                "Instructions Time": [data["instructions"]["timeelapsed"]],
                "Block 1 Time": [data["block_one"]["timeelapsed"]],
                "Block 2 Time": [data["block_two"]["timeelapsed"]],
                "Block 3 Time": [data["block_three"]["timeelapsed"]],
                "Final Score": [data["block_three"]["score"]],
            }
        )

        df_summary = pd.concat([subj_df_summary, df_summary], ignore_index=True)

        block_one_df = pd.DataFrame.from_dict(
            {
                "PID": [data["block_one"]["participant_id"]] * n_trials,
                "Run": [data["instructions"]["run"][-1]] * n_trials,
                "Block": [data["block_one"]["phase"]] * n_trials,
                "Type": [data["block_one"]["block"]] * n_trials,
                "Trial": np.arange(n_trials),
                "Action": data["block_one"]["choices"],
                "Reward": data["block_one"]["outcomes"],
                "RT": np.array(data["block_one"]["choicetimes"])
                - np.array(data["block_one"]["cuetimes"]),
            }
        )
        if (
            subj_df_summary["Reversal Timings"][0] == "C"
            or subj_df_summary["Reversal Timings"][0] == "D"
        ):
            block_one_df["Action"] = 1 - block_one_df["Action"]

        block_two_df = pd.DataFrame.from_dict(
            {
                "PID": [data["block_two"]["participant_id"]] * n_trials,
                "Run": [data["instructions"]["run"][-1]] * n_trials,
                "Block": [data["block_two"]["phase"]] * n_trials,
                "Type": [data["block_two"]["block"]] * n_trials,
                "Trial": np.arange(n_trials),
                "Action": data["block_two"]["choices"],
                "Reward": data["block_two"]["outcomes"],
                "RT": np.array(data["block_two"]["choicetimes"])
                - np.array(data["block_two"]["cuetimes"]),
            }
        )
        if (
            subj_df_summary["Reversal Timings"][0] == "C"
            or subj_df_summary["Reversal Timings"][0] == "D"
        ):
            block_two_df["Action"] = 1 - block_two_df["Action"]

        block_three_df = pd.DataFrame.from_dict(
            {
                "PID": [data["block_three"]["participant_id"]] * n_trials,
                "Run": [data["instructions"]["run"][-1]] * n_trials,
                "Block": [data["block_three"]["phase"]] * n_trials,
                "Type": [data["block_three"]["block"]] * n_trials,
                "Trial": np.arange(n_trials),
                "Action": data["block_three"]["choices"],
                "Reward": data["block_three"]["outcomes"],
                "RT": np.array(data["block_three"]["choicetimes"])
                - np.array(data["block_three"]["cuetimes"]),
            }
        )
        if (
            subj_df_summary["Reversal Timings"][0] == "C"
            or subj_df_summary["Reversal Timings"][0] == "D"
        ):
            block_three_df["Action"] = 1 - block_three_df["Action"]

        subj_longform = pd.concat(
            [block_one_df, block_two_df, block_three_df], ignore_index=True
        )
        longform = pd.concat([subj_longform, longform], ignore_index=True)
        longform = (
            longform.replace("numberbar_pos", "positive")
            .replace("numberbar_neg", "negative")
            .replace("numberbar_mixed", "mixed")
        )

    return df_summary, longform

    # Create dictionary of data
    # bad_inds = []
    # act_rew_rate = {
    #     "pids": [],
    #     "reversal_timings": [],
    #     "block_order": [],
    #     "final_score": [],
    #     "positive": {
    #         "block_num": np.zeros(len(li)),
    #         "actions": np.zeros((len(li), num_trials)),
    #         "rewards": np.zeros((len(li), num_trials)),
    #         "rts": np.zeros((len(li), num_trials)),
    #         "timeelapsed": np.zeros(len(li)),
    #     },
    #     "negative": {
    #         "block_num": np.zeros(len(li)),
    #         "actions": np.zeros((len(li), num_trials)),
    #         "rewards": np.zeros((len(li), num_trials)),
    #         "rts": np.zeros((len(li), num_trials)),
    #         "timeelapsed": np.zeros(len(li)),
    #     },
    #     "mixed": {
    #         "block_num": np.zeros(len(li)),
    #         "actions": np.zeros((len(li), num_trials)),
    #         "rewards": np.zeros((len(li), num_trials)),
    #         "rts": np.zeros((len(li), num_trials)),
    #         "timeelapsed": np.zeros(len(li)),
    #     },
    # }

    # for i, subject in enumerate(li):
    #     # Obtain PIDs
    #     instructions_data = subject[subject["phase"] == "instructions"]
    #     block_order = np.array(eval(instructions_data["block_order"].to_numpy()[0]))
    #     act_rew_rate["block_order"].append(block_order)
    #     act_rew_rate["pids"].append(instructions_data["participant_id"].to_numpy()[0])
    #     act_rew_rate["reversal_timings"].append(
    #         instructions_data["reversal_timings"][0]
    #     )

    #     for j, num_block in enumerate(["block_one", "block_two", "block_three"]):

    #         block_data = subject[subject["phase"] == num_block]
    #         if num_block == "block_three":
    #             act_rew_rate["final_score"].append(block_data["score"].to_numpy()[0])
    #         block_id = block_data["block"].to_numpy()[0]
    #         if block_id == "numberbar_pos":
    #             block = "positive"
    #         elif block_id == "numberbar_neg":
    #             block = "negative"
    #         if block_id == "numberbar_mixed":
    #             block = "mixed"
    #         act_rew_rate[block]["block_num"][i] = j + 1

    #         # Set timeelapsed
    #         act_rew_rate[block]["timeelapsed"][i] = block_data[
    #             "timeelapsed"
    #         ].to_numpy()[0]

    #         # Obtain choices
    #         choices = np.array(eval(block_data["choices"].to_numpy()[0]))

    #         if (
    #             instructions_data["reversal_timings"][0] == "C"
    #             or instructions_data["reversal_timings"][0] == "D"
    #         ):
    #             choices = 1 - choices
    #         act_rew_rate[block]["actions"][i, :] = choices

    #         # Obtain rewards
    #         rewards = np.array(eval(block_data["outcomes"].to_numpy()[0]))
    #         act_rew_rate[block]["rewards"][i, :] = np.array(rewards)

    #         # Obtain money RTs
    #         cuetimes = np.array(eval(block_data["cuetimes"].to_numpy()[0]))
    #         choicetimes = np.array(eval(block_data["choicetimes"].to_numpy()[0]))
    #         act_rew_rate[block]["rts"][i, :] = choicetimes - cuetimes

    # act_rew_rate["pids"] = np.array(act_rew_rate["pids"])
    # act_rew_rate["final_score"] = np.array(act_rew_rate["final_score"])
    # act_rew_rate["block_order"] = np.array(act_rew_rate["block_order"])
    # act_rew_rate["reversal_timings"] = np.array(act_rew_rate["reversal_timings"])
    # for block in ["positive", "negative", "mixed"]:
    #     act_rew_rate[block]["actions"] = act_rew_rate[block]["actions"].astype(int)

    # # if remove_high_rt:
    # #     print('removing high rts')
    # #     for i, subject in enumerate(act_rew_rate['pids']):
    # #         max_money_rt = act_rew_rate['money']['rts'][i].max()
    # #         max_other_rt = act_rew_rate['other']['rts'][i].max()
    # #         if max_money_rt > 10 or max_other_rt > 10:
    # #             print(i, act_rew_rate['pids'][i])
    # #             bad_inds.append(i)
    # #     print(bad_inds)

    # if remove_below_chance:
    #     print("removing below chance")
    #     for i, subject in enumerate(act_rew_rate["pids"]):
    #         total_below_chance = 0

    #         if (
    #             act_rew_rate["reversal_timings"][i] == "A"
    #             or act_rew_rate["reversal_timings"][i] == "C"
    #         ):
    #             timings = np.array([12, 12, 11])
    #         elif (
    #             act_rew_rate["reversal_timings"][i] == "B"
    #             or act_rew_rate["reversal_timings"][i] == "D"
    #         ):
    #             timings = np.array([13, 12, 10])

    #         reward_structure = np.array(
    #             [
    #                 np.array(
    #                     [0.8] * timings[0] + [0.2] * timings[1] + [0.8] * timings[2]
    #                 ),
    #                 np.array(
    #                     [0.2] * timings[0] + [0.8] * timings[1] + [0.2] * timings[2]
    #                 ),
    #             ]
    #         )

    #         for j, block in enumerate(["block_one", "block_two", "block_three"]):
    #             if act_rew_rate["positive"]["block_num"][i] == j + 1:
    #                 choices = act_rew_rate["positive"]["actions"][i]
    #             elif act_rew_rate["negative"]["block_num"][i] == j + 1:
    #                 choices = act_rew_rate["negative"]["actions"][i]
    #             elif act_rew_rate["mixed"]["block_num"][i] == j + 1:
    #                 choices = act_rew_rate["mixed"]["actions"][i]

    #             optimal_choices = np.array(
    #                 reward_structure[0, :] < reward_structure[1, :]
    #             ).astype(int)
    #             if block == "block_two":
    #                 optimal_choices = 1 - optimal_choices

    #             acc = (
    #                 np.sum((np.array(choices) == optimal_choices).astype(int))
    #                 / num_trials
    #             )

    #             num_choice_changes = (np.diff(choices) != 0).sum()

    #             if acc < remove_below_chance or num_choice_changes < 2:
    #                 # if acc < remove_below_chance:
    #                 print(i, act_rew_rate["pids"][i], block)
    #                 # bad_inds.append(i)
    #                 total_below_chance += 1

    #         if total_below_chance >= 1:
    #             bad_inds.append(i)
    #     print(set(bad_inds))

    # print("all good inds")
    # good_inds = np.setdiff1d(np.arange(len(act_rew_rate["pids"])), bad_inds)
    # print(good_inds)
    # act_rew_rate["pids"] = act_rew_rate["pids"][good_inds]
    # act_rew_rate["reversal_timings"] = act_rew_rate["reversal_timings"][good_inds]
    # act_rew_rate["block_order"] = act_rew_rate["block_order"][good_inds]
    # for b in ["positive", "negative", "mixed"]:
    #     act_rew_rate[b]["block_num"] = act_rew_rate[b]["block_num"][good_inds]
    #     act_rew_rate[b]["actions"] = act_rew_rate[b]["actions"][good_inds]
    #     act_rew_rate[b]["rewards"] = act_rew_rate[b]["rewards"][good_inds]
    #     act_rew_rate[b]["rts"] = act_rew_rate[b]["rts"][good_inds]

    # return act_rew_rate, np.array(good_inds)


def load_data(
    path_to_data, num_trials=35, remove_below_chance=False, remove_high_rt=False
):

    # Load all csvs into li
    all_csvs = glob.glob(path_to_data + "*.csv")
    li = []
    for filename in all_csvs:
        df = pd.read_csv(filename)
        li.append(df)

    # Create dictionary of data
    bad_inds = []
    act_rew_rate = {
        "pids": [],
        "reversal_timings": [],
        "block_order": [],
        "final_score": [],
        "positive": {
            "block_num": np.zeros(len(li)),
            "actions": np.zeros((len(li), num_trials)),
            "rewards": np.zeros((len(li), num_trials)),
            "rts": np.zeros((len(li), num_trials)),
            "timeelapsed": np.zeros(len(li)),
        },
        "negative": {
            "block_num": np.zeros(len(li)),
            "actions": np.zeros((len(li), num_trials)),
            "rewards": np.zeros((len(li), num_trials)),
            "rts": np.zeros((len(li), num_trials)),
            "timeelapsed": np.zeros(len(li)),
        },
        "mixed": {
            "block_num": np.zeros(len(li)),
            "actions": np.zeros((len(li), num_trials)),
            "rewards": np.zeros((len(li), num_trials)),
            "rts": np.zeros((len(li), num_trials)),
            "timeelapsed": np.zeros(len(li)),
        },
    }

    for i, subject in enumerate(li):
        # Obtain PIDs
        instructions_data = subject[subject["phase"] == "instructions"]
        block_order = np.array(eval(instructions_data["block_order"].to_numpy()[0]))
        act_rew_rate["block_order"].append(block_order)
        act_rew_rate["pids"].append(instructions_data["participant_id"].to_numpy()[0])
        act_rew_rate["reversal_timings"].append(
            instructions_data["reversal_timings"][0]
        )

        for j, num_block in enumerate(["block_one", "block_two", "block_three"]):

            block_data = subject[subject["phase"] == num_block]
            if num_block == "block_three":
                act_rew_rate["final_score"].append(block_data["score"].to_numpy()[0])
            block_id = block_data["block"].to_numpy()[0]
            if block_id == "numberbar_pos":
                block = "positive"
            elif block_id == "numberbar_neg":
                block = "negative"
            if block_id == "numberbar_mixed":
                block = "mixed"
            act_rew_rate[block]["block_num"][i] = j + 1

            # Set timeelapsed
            act_rew_rate[block]["timeelapsed"][i] = block_data[
                "timeelapsed"
            ].to_numpy()[0]

            # Obtain choices
            choices = np.array(eval(block_data["choices"].to_numpy()[0]))

            if (
                instructions_data["reversal_timings"][0] == "C"
                or instructions_data["reversal_timings"][0] == "D"
            ):
                choices = 1 - choices
            act_rew_rate[block]["actions"][i, :] = choices

            # Obtain rewards
            rewards = np.array(eval(block_data["outcomes"].to_numpy()[0]))
            act_rew_rate[block]["rewards"][i, :] = np.array(rewards)

            # Obtain money RTs
            cuetimes = np.array(eval(block_data["cuetimes"].to_numpy()[0]))
            choicetimes = np.array(eval(block_data["choicetimes"].to_numpy()[0]))
            act_rew_rate[block]["rts"][i, :] = choicetimes - cuetimes

    act_rew_rate["pids"] = np.array(act_rew_rate["pids"])
    act_rew_rate["final_score"] = np.array(act_rew_rate["final_score"])
    act_rew_rate["block_order"] = np.array(act_rew_rate["block_order"])
    act_rew_rate["reversal_timings"] = np.array(act_rew_rate["reversal_timings"])
    for block in ["positive", "negative", "mixed"]:
        act_rew_rate[block]["actions"] = act_rew_rate[block]["actions"].astype(int)

    # if remove_high_rt:
    #     print('removing high rts')
    #     for i, subject in enumerate(act_rew_rate['pids']):
    #         max_money_rt = act_rew_rate['money']['rts'][i].max()
    #         max_other_rt = act_rew_rate['other']['rts'][i].max()
    #         if max_money_rt > 10 or max_other_rt > 10:
    #             print(i, act_rew_rate['pids'][i])
    #             bad_inds.append(i)
    #     print(bad_inds)

    if remove_below_chance:
        print("removing below chance")
        for i, subject in enumerate(act_rew_rate["pids"]):
            total_below_chance = 0

            if (
                act_rew_rate["reversal_timings"][i] == "A"
                or act_rew_rate["reversal_timings"][i] == "C"
            ):
                timings = np.array([12, 12, 11])
            elif (
                act_rew_rate["reversal_timings"][i] == "B"
                or act_rew_rate["reversal_timings"][i] == "D"
            ):
                timings = np.array([13, 12, 10])

            reward_structure = np.array(
                [
                    np.array(
                        [0.8] * timings[0] + [0.2] * timings[1] + [0.8] * timings[2]
                    ),
                    np.array(
                        [0.2] * timings[0] + [0.8] * timings[1] + [0.2] * timings[2]
                    ),
                ]
            )

            for j, block in enumerate(["block_one", "block_two", "block_three"]):
                if act_rew_rate["positive"]["block_num"][i] == j + 1:
                    choices = act_rew_rate["positive"]["actions"][i]
                elif act_rew_rate["negative"]["block_num"][i] == j + 1:
                    choices = act_rew_rate["negative"]["actions"][i]
                elif act_rew_rate["mixed"]["block_num"][i] == j + 1:
                    choices = act_rew_rate["mixed"]["actions"][i]

                optimal_choices = np.array(
                    reward_structure[0, :] < reward_structure[1, :]
                ).astype(int)
                if block == "block_two":
                    optimal_choices = 1 - optimal_choices

                acc = (
                    np.sum((np.array(choices) == optimal_choices).astype(int))
                    / num_trials
                )

                num_choice_changes = (np.diff(choices) != 0).sum()

                if acc < remove_below_chance or num_choice_changes < 2:
                    # if acc < remove_below_chance:
                    print(i, act_rew_rate["pids"][i], block)
                    # bad_inds.append(i)
                    total_below_chance += 1

            if total_below_chance >= 1:
                bad_inds.append(i)
        print(set(bad_inds))

    print("all good inds")
    good_inds = np.setdiff1d(np.arange(len(act_rew_rate["pids"])), bad_inds)
    print(good_inds)
    act_rew_rate["pids"] = act_rew_rate["pids"][good_inds]
    act_rew_rate["reversal_timings"] = act_rew_rate["reversal_timings"][good_inds]
    act_rew_rate["block_order"] = act_rew_rate["block_order"][good_inds]
    for b in ["positive", "negative", "mixed"]:
        act_rew_rate[b]["block_num"] = act_rew_rate[b]["block_num"][good_inds]
        act_rew_rate[b]["actions"] = act_rew_rate[b]["actions"][good_inds]
        act_rew_rate[b]["rewards"] = act_rew_rate[b]["rewards"][good_inds]
        act_rew_rate[b]["rts"] = act_rew_rate[b]["rts"][good_inds]

    return act_rew_rate, np.array(good_inds)


def subset(orig_act_rew_rate, subset_ind):

    act_rew_rate = copy.deepcopy(orig_act_rew_rate)

    act_rew_rate["pids"] = act_rew_rate["pids"][subset_ind]

    act_rew_rate["money"]["actions"] = act_rew_rate["money"]["actions"][subset_ind, :]
    act_rew_rate["money"]["rewards"] = act_rew_rate["money"]["rewards"][subset_ind, :]
    act_rew_rate["money"]["ratings"] = act_rew_rate["money"]["ratings"][subset_ind, :]

    act_rew_rate["other"]["actions"] = act_rew_rate["other"]["actions"][subset_ind, :]
    act_rew_rate["other"]["rewards"] = act_rew_rate["other"]["rewards"][subset_ind, :]
    act_rew_rate["other"]["ratings"] = act_rew_rate["other"]["ratings"][subset_ind, :]

    return act_rew_rate


def load_questionnaire_data(path_to_qdata, pids, fields=None):
    redcap_csv_path = glob.glob(path_to_qdata)[0]
    redcap_df = pd.read_csv(redcap_csv_path)

    if fields is None:
        fields = [
            "prolific_pid",
            "gi_1",
            "gi_2",
            "mshq_1",
            "mshq_2",
            "mshq_3",
            "mshq_4",
            "mshq_5",
            "mshq_6",
            "mshq_7",
            "mshq_8",
            "mshq_9",
            "mshq_10",
            "mshq_11",
            "mshq_12",
            "mshq_13",
            "mshq_14",
            "mshq_15",
            "mshq_16",
            "mshq_17",
            "mshq_18",
            "mshq_19",
            "mshq_20",
            "mshq_21",
            "mps_1",
            "mps_2",
            "mps_3",
            "mps_4",
            "mps_5",
            "mps_6",
            "mps_7",
            "mps_8",
            "mps_9",
            "mps_10",
            "mps_11",
            "mps_12",
            "mps_13",
            "mps_14",
            "mps_15",
            "mps_16",
            "mps_17",
            "mps_18",
            "mps_19",
        ]

    redcap_df = redcap_df.filter(items=fields)

    valid_redcap_df = redcap_df[redcap_df["prolific_pid"].isin(pids)]
    valid_redcap_df = valid_redcap_df.set_index("prolific_pid").loc[pids]

    return valid_redcap_df
