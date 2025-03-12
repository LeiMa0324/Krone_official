
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class IaasProcessor:

    def __init__(self, structure_logs,
                 flow_configs,
                 rollback_flow_only=False,
                 truncate_before_rollback=False,
                 save_normal_after_truncation=False,
                 slide=False,
                 slide_by='logkey',
                 structured_process =None,
                 window_size= 10,
                 window_step = 5,
                 ):
        self.logs = structure_logs
        self.flow_configs = flow_configs
        self.flow_names = self.flow_configs["flow_names"]
        self.anomaly_ratios = {self.flow_names[i]: self.flow_configs["anomaly_ratios"][i] for i in range(0, len(self.flow_names))}
        self.flow_normal_sizes = {self.flow_names[i]: self.flow_configs["normal_sizes"][i] for i in
                              range(0, len(self.flow_names))}
        self.truncate_before_rollback = truncate_before_rollback
        self.save_normal_after_truncation = save_normal_after_truncation
        self.structured_processes = structured_process

        self.slide = slide
        self.slide_by = slide_by
        self.window_size = window_size
        self.window_step = window_step

        if self.slide_by == 'entity' and structured_process is None:
            raise Exception("Please specify the structured_process file with entity information if sliding by entity!")

        self.log_sequence_df = None


    def run(self,  output_dir = '', output_filename = 'Iaas_sequences.csv'):
        self.filter_by_flow_name() # select the flows
        self._fill_flow_name_col() # find the flow name for rows with empty value and fill them
        if self.truncate_before_rollback:  # for abnormal flows: removes the logs after the rollback action
            self._truncate_before_rollback()

        if self.slide:
            raise NotImplementedError()
        else:
            sequence_df = self.partition_by_flow() # partition sequences by flow id

            flow_datas = []
            for flow_name in sequence_df["flow_name"].unique():
                # for each type of flow, select anomaly ratio of anomalies
                flow_data = sequence_df[sequence_df["flow_name"]==flow_name]
                flow_normal = flow_data[flow_data["Label"]==0]
                flow_normal_size = len(flow_normal)

                duplicate_normal_sequences = flow_normal["EventSequence"].value_counts()
                duplicate_normal_sequences = duplicate_normal_sequences[
                    duplicate_normal_sequences > 1].index  # Select sequences appearing more than once

                # Select rows where EventSequence appears more than once
                duplicate_normal_rows = flow_normal[flow_normal["EventSequence"].isin(duplicate_normal_sequences)]
                unduplicate_normal_rows = flow_normal[~flow_normal["EventSequence"].isin(duplicate_normal_sequences)]

                if isinstance(self.flow_normal_sizes[flow_name], float) and self.flow_normal_sizes[flow_name] <= 1.0:
                    selected_flow_normal_size = int(flow_normal_size*self.flow_normal_sizes[flow_name])
                elif isinstance(self.flow_normal_sizes[flow_name], int) and self.flow_normal_sizes[flow_name] > 1:
                    selected_flow_normal_size = self.flow_normal_sizes[flow_name]
                else:
                    raise Exception(f"unsupported flow_name type {type(self.flow_normal_sizes[flow_name])}")

                if selected_flow_normal_size > len(duplicate_normal_rows):
                    flow_normal_unduplicate = unduplicate_normal_rows.sample(n=selected_flow_normal_size-len(duplicate_normal_rows))
                    flow_normal = pd.concat([flow_normal_unduplicate, duplicate_normal_rows])
                else:
                    flow_normal = duplicate_normal_rows.sample(n=selected_flow_normal_size)

                flow_abnormal = flow_data[flow_data["Label"]==1]
                flow_anomaly_ratio = self.anomaly_ratios[flow_name]

                flow_selected_abnormal = flow_abnormal.sample(n = int(len(flow_normal)* flow_anomaly_ratio))
                flow_sequence_df = pd.concat([flow_normal, flow_selected_abnormal])
                flow_datas.append(flow_sequence_df)

                sequence_df = pd.concat(flow_datas)
                sequence_df.reset_index()

            sequence_df["seq_id"] = sequence_df.index
            sequence_df = sequence_df[["seq_id", "flow_id","flow_name","Label","EventSequence"]]
            if not os.path.exists(output_dir) and output_dir != '':
                os.makedirs(output_dir)
            sequence_df.to_csv(os.path.join(output_dir, output_filename), index=False)
            readme = self._sequence_statistics(sequence_df)
            with open(os.path.join(output_dir, 'Iaas_data_readme.txt'), 'w') as f:
                f.write(readme)

    def _fill_flow_name_col(self):
        if self.logs["flow_name"].isna().any():
            flow_id_to_flow_name = {}
            for flow_id in tqdm(self.logs["flow_id"].unique()):
                flow = self.logs[self.logs["flow_id"] == flow_id]
                flow_name = list(flow["flow_name"].dropna().unique())[0]
                flow_id_to_flow_name[flow_id] = flow_name

            self.logs["flow_name"] = self.logs["flow_id"].map(flow_id_to_flow_name)

    def filter_by_flow_name(self):
        ori_len = len(self.logs)
        self.logs = self.logs.loc[self.logs["flow_name"].isin(self.flow_names)]
        print(f"Original logs: {ori_len}, Filtered logs: {len(self.logs)}")

    def _truncate_before_rollback(self):
        print("Starting to truncate before rollback action...")
        count = 0
        truncated_logs = []
        for flow_id in tqdm(self.logs["flow_id"].unique()):
            flow = self.logs[self.logs["flow_id"] == flow_id]
            flow = flow.sort_values(by=['T'])
            label = flow["label"].tolist()[0]
            is_roll_back = label == 1 and flow["action"].isin(['rollback']).any()
            if  is_roll_back:
                rollback_time = flow[flow["action"] =='rollback'].iloc[0]['T']
                if self.save_normal_after_truncation:
                    normal_flow = flow[flow["T"] >= rollback_time]
                    normal_flow = normal_flow.sort_values(by=['T', 'EventId'])
                    normal_flow["flow_id"] = flow_id+"_truncatedNormal"
                    normal_flow["label"]=0
                    truncated_logs.append(normal_flow)
                flow = flow[flow["T"] < rollback_time]
                count +=1
            flow = flow.sort_values(by=['T', 'EventId'])
            truncated_logs.append(flow)

        print(f"{count} sequence truncated before rollback")
        self.logs = pd.concat(truncated_logs)

    def partition_by_flow(self):
        print("Starting to partition logs by flow id...")

        sequence_df = {"flow_id": [], "flow_name": [], "EventSequence": [], "Label": []}

        for flow_id in tqdm(self.logs["flow_id"].unique()):

            flow = self.logs[self.logs["flow_id"] == flow_id]
            flow = flow.sort_values(by=['T'])

            label = flow["label"].tolist()[0]
            flow = flow.sort_values(by=['T', 'EventId'])
            sequence = "["+','.join([str(e) for e in flow["EventId"].tolist()])+"]"
            flow_name = list(flow["flow_name"].dropna().unique())[0]

            sequence_df["flow_id"].append(flow_id)
            sequence_df["flow_name"].append(flow_name)
            sequence_df["EventSequence"].append(sequence)
            sequence_df["Label"].append(label)

        sequence_df = pd.DataFrame(sequence_df)

        print("Partition finished.")
        return sequence_df

    def _partition_by_flow_slide_by_entity(self, entity_window_size, entity_step_size, if_truncate_rollback=False):
        print(f"Starting to partition logs by flow id and slide by entity window size {entity_window_size} and slide by entity step size {entity_step_size}...")
        if "entity_1" not in self.logs.columns:
            self.logs["entity_1"] = self.logs["EventId"].map(
                lambda e: self.structured_processes[self.structured_processes["event_id"] == e]["entity_1"].iloc[0])

        sequence_df = {"flow_id": [], "flow_name": [], "EventSequence": [], "Label": []}

        for flow_id in tqdm(self.logs["flow_id"].unique()):
            flow = self.logs[self.logs["flow_id"] == flow_id]
            flow = flow.sort_values(by=['T'])
            flow = flow.sort_values(by=['T', 'EventId'])
            flow_name = list(flow["flow_name"].dropna().unique())[0]

            entities = flow["entity_1"]
            last_entity = entities.iloc[0]
            last_entity_start_idx = 0
            last_entity_end_idx = 0
            entity_df = {"entity": [], "start_idx": [], "end_idx": []}

            for id, entity in enumerate(entities):
                if entity == last_entity:
                    last_entity_end_idx = id
                else:
                    entity_df["entity"].append(last_entity)
                    entity_df["start_idx"].append(last_entity_start_idx)
                    entity_df["end_idx"].append(last_entity_end_idx)

                    last_entity = entity
                    last_entity_start_idx = id
                    last_entity_end_idx = id

            entity_df = pd.DataFrame(entity_df)

            seq_indices = []
            for i in range(0, len(entity_df) - entity_window_size + 1, entity_step_size):
                entity_seq = entity_df[i:i + entity_window_size]
                entity_seq_start_idx = entity_seq["start_idx"].min()
                entity_seq_end_idx = entity_seq["end_idx"].max() + 1
                seq_indices.append((entity_seq_start_idx, entity_seq_end_idx))
            #
            for j, window in enumerate(seq_indices):
                if window[1] < len(flow):
                    assert flow.iloc[window[1]]["entity_1"] != flow.iloc[window[1] - 1]["entity_1"]
                if window[0] > 0:
                    assert flow.iloc[window[0] - 1]["entity_1"] != flow.iloc[window[0]]["entity_1"]

                label = flow[window[0]: window[1]]["label"].max()
                sequence = "[" + ",".join([str(e) for e in flow[window[0]: window[1]]["EventId"].tolist()]) + "]"
                sequence_df["flow_id"].append(flow_id)
                sequence_df["flow_name"].append(flow_name)
                sequence_df["EventSequence"].append(sequence)
                sequence_df["Label"].append(label)


        sequence_df = pd.DataFrame(sequence_df)
        # correct the wrongly anomalies due to sliding windows
        for seq in sequence_df["EventSequence"].unique():
            idx = sequence_df["EventSequence"] == seq
            seq_df = sequence_df[idx]
            if len(seq_df["Label"].unique()) > 1:
                sequence_df.loc[idx, "Label"] = 0
        sequence_df["seq_id"] = sequence_df.index
        print(f"Partition finished.")
        self._sequence_statistics(sequence_df)
        return sequence_df

    def _sequence_statistics(self, sequence_df):
        readme ='Configs:\n'
        readme += (f'   flow names: {self.flow_configs}\n'
                   f'   rollback_flow_only: {self.rollback_flow_only}\n'
                   f'   truncate_before_rollback: {self.truncate_before_rollback}\n'
                   f'   save_normal_after_truncation: {self.save_normal_after_truncation}\n'
                  f'    Slide:{self.slide}\n')
        if self.slide:
            readme += (f"   Slide by: {self.slide_by}\n"
                       f"   Window size: {self.window_size}\n"
                       f"   Window step: {self.window_step}\n")
        total = len(sequence_df)
        normal = sequence_df[sequence_df["Label"] ==0]
        abnormal = sequence_df[sequence_df["Label"] ==1]
        abnormal_ratio = round(len(abnormal) / total, 4)
        unique_sequence = len(sequence_df["EventSequence"].unique())

        readme += 'Overall Statistics:\n'
        readme +=(f"    Total sequence: {total}\n")
        readme +=(f"    Normal sequence: {len(normal)}, unqiue normal sequences: {len(normal['EventSequence'].unique())}\n")
        readme +=(f"    Abnormal sequence: {len(abnormal)}, unqiue abnormal sequences: {len(abnormal['EventSequence'].unique())}\n")
        readme +=(f"    Anomaly ratio: {abnormal_ratio}\n")
        readme +=(f"    Unique sequence: {unique_sequence}\n")
        return readme

logs = pd.read_csv("../data/Iaas/Iaas.log_structured.csv") # structured logs

flow_configs = {"flow_names":["CreateInstance", "DetachNetworkInterface", "AttachNetworkInterface"],
                "normal_sizes": [1500, 1.0, 1.0],
                "anomaly_ratios": [0.2, 1.0, 1.0],}

#
# flow_configs = {"flow_names":["DetachNetworkInterface", "AttachNetworkInterface"],
#                 "normal_sizes": [ 1.0, 1.0],
#                 "anomaly_ratios": [ 1.0, 1.0],}


processor = IaasProcessor(structure_logs=logs,
flow_configs = flow_configs,
                          rollback_flow_only=False,
                          truncate_before_rollback=True, # extract the sequence before the roll back action for abnormal flows, default yes
                          save_normal_after_truncation=True,
                          slide=False) # if do sliding window, recommend no
processor.run(output_dir='../data/IaaS_v13', output_filename='IaaS_v13_sequences.csv') #数据集名字Iaa
# anomaly_ratio: the raio between selected anomaly/normal