ENTITY_MERGE_PROMPT_HARD = """
-Goal-
Given {process_num} system processes corresponding to an execution workflow, find an execution-level ENTITY word for the whole execution, which is in one of the following types: [{entity_types}]

-Input-
Each process consists of a short description, a process-level ENTITY that may initialize or undergo the process as reference. The processes are given in the following format:
Process 1: <process_1_desc> | <process_entity_of_process_1>
Process 2: <process_2_desc> | <process_entity_of_process_2>
...
Process {process_num}: <process_{process_num}_desc> | <process_entity_of_process_{process_num}>

-Output-
1. By examining the processes and verifying the given process-level entities, find the execution-level ENTITY that best represents all the processes.
2. If the execution-level entity contains multiple words, make it in camel case without blank space.
3. Return the output in the following json format:
{"execution_entity": <entity for the execution, please return the string surrounded by double quotes.>,
}


######################
-Real Data-
######################
{process_list}
######################
Output:"""

ENTITY_MERGE_PROMPT_SOFT = """
-Goal-
Given {process_num} text documents that each one is potentially relevant to a system process, try to decide if the all processes have the same ENTITY. 

-Input-
Each process consists of a short description, a trial ENTITY that may initialize or undergo the process as reference, which is in one of the following types: [{entity_types}]. The processes will be given in the following format:
Process 1: <process_1_desc> | <trial_entity_of_process_1>
Process 2: <process_2_desc> | <trial_entity_of_process_2>
...
Process {process_num}: <process_{process_num}_desc> | <entity_of_process_{process_num}>

-Steps-
1. By examining the processes and verifying the given trial entities, decide if these processes have the same actual ENTITY, the decision is True if they have the same actual entity, False otherwise. 
2. If they do have the same actual ENTITY, return the ENTITY. It can be selected from the given trial ENTITIES or generated if none of the given ones is suitable. If generated, the type of the entity should be one of the following types: [{entity_types}]
3. Mark the ENTITY source as "selected" if it is selected from the given trial ENTITIES, or "generated" if it is generated.
Return the output in the following json format:
{"decision": <"True" or "False", please return the string surrounded by double quotes.>,
"entity": <the common entity, make it "none" if the decision is "False", please return the string surrounded by double quotes.>
"entity_source": <"selected" or "generated", make it "none" if the decision is "False", please return the string surrounded by double quotes.>}


######################
-Real Data-
######################
{process_list}
######################
Output:"""

ACTION_MERGE_PROMPT_SOFT = """
-Goal-
Given {process_num} text documents that each one is potentially relevant to a system process, try to decide if the all processes have the same ACTION. 

-Input-
Each process consists of a short description, an ENTITY of the process and a trial ACTION possibly related to the ENTITY. The processes will be given in the following format:
Process 1: <process_1_desc> | <entity_of_process_1> | <trial_action_of_process_1>
Process 2: <process_2_desc> | <entity_of_process_2> | <trial_action_of_process_2>
...
Process {process_num}: <process_{process_num}_desc> | <entity_of_process_{process_num}> | <trial_action_of_process_{process_num}>
Note that the all processes always have the same entity.

-Steps-
1. By examining the processes and verifying the given trial actions, decide if these processes have the same actual ACTION related to the common ENTITY, the decision is "True" if they have the same action, "False" otherwise. 
2. If they do have the same actual ACTION, return the action. It can be selected from the given trial actions or generated if none of the given ones is suitable.
3. Mark the action source as "selected" if it is selected from the given trial actions, or "generated" if it is generated.
Return the output in the following json format:
{"decision": <"True" or "False", please return the string surrounded by double quotes.>,
"action": <the common action, make it "none" if the decision is False, please return the string surrounded by double quotes.>
"action_source": <"selected" or "generated", make it "none" if the decision is "False", please return the string surrounded by double quotes.>}


######################
-Real Data-
######################
{process_list}
######################
Output:"""

ENTITY_SUMMARY_PROMPT = """
-Role-
You are an expert in understanding and summarizing system execution. Given description of some relevant execution processes, you are good at summarizing them  in respect to their common {entity_type} entity.

-Goal- 
Given a list of execution processes of the same {entity_type} entity as the context, generate a summary for the {entity_type} entity about its relevant execution processes. Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the processes.

-Input-
The input includes a list of execution processes, each process is consisted of the following part: 
- entity: an entity of the {entity_type} type
- action: refers to any specific operation, task, or behavior that is either performed by the entity itself or enacted upon it.
- status: refers to the condition or state of the entity following the completion of an action.
- description: a short text description of the process
If the entity, action or status contains "none", pay more attention to the description. 

Each process follow the following format:
Process: entity: <entity> {DELIMITER} entity_type: <entity_type> {DELIMITER} action: <action> {DELIMITER} status: <status> {DELIMITER} description: <description> {PROCESS_DELIMITER}

-Output-
Generate the summary of the entity in respect to the execution.

######################
-Real Data-
######################
-Input-
{input_processes}
######################
Output:"""

ENTITY_PATH_SUMMARY_PROMPT = """
-Role-
You are an expert in understanding and summarizing the sequential workflow of system execution. 

-Goal- 
Given a sequence of entities which represents an execution workflow. Each entity corresponds to a specific system module or object, encompassing multiple distinct processes. Please summarize the entity sequence into a single, comprehensive description in respect to the execution flow. Make sure to include information collected from all the entities.

-Input
The input includes an entity sequence and the description of each unique entity in the following format: 
entity sequence: <entity_1> {DELIMETER} <entity_2> {DELIMETER} ... <entity_n>
entity description: 
    <entity_1>: <entity_1_desc>
    <entity_2>: <entity_1_desc>
    ...
    <entity_k>: <entity_n_desc>

-Output-
Generate the summary of the execution represented by the entity sequence.

######################
-Real Data-
######################
-Input-
Entity sequence: {input_entity_seq}
Entity description: 
{input_entity_desc}
######################
Output:"""


SEQUENCE_DETECT_PROMPT = """
-Role-
You are an expert in understanding normal_flows execution workflows as well as detecting abnormal_flows execution workflows. 

-Goal- 
Given several sequences of entities which represents normal_flows execution workflows as the context, decide if a test entity sequence indicates an abnormal_flows execution workflow. 

-Context-
The context includes the following content:
    - k (k = {example_num}) examples of entity sequences which represents normal_flows execution workflows. Each example is consisted of the entity sequence and a text description of the sequence and the execution workflow.
    - The descriptions of all m (m = {entity_num}) unique entities in the test sequence.
The context follows the following format:
Example 1 - Entity sequence: <entity sequence 1>
Example 1 - Description: <text description of entity sequence 1>
...
Example k - Entity sequence: <entity sequence k>
Example k - Description: <text description of entity sequence k>
Entity <entity_1>: <entity_1_desc>
...
Entity <entity_m>: <entity_m_desc>

-Input-
<test entity sequence>

-Output-
Return the prediction and reason of the prediction for the test sequence
- prediction: Return "Normal" or "Abnormal" as the decision for the test entity sequence.
- reason: Return the reason for the prediction
Generate the output in the following format:
{
"prediction": <prediction>,
"reason": <reason>
}


######################
-Real Data-
######################
-Context-
{context_examples}

-Input-
{test_entity_seq}
######################
Output:"""


PROCESS_SEQ_SUMMARY_PROMPT_V1 = """
-Role-
You are an expert in understanding and summarizing the sequential workflow of system execution. 

-Goal- 
Given a sequence of PROCESSES which represents an execution workflow. Please summarize the PROCESS sequence into a single, comprehensive description in respect to the execution flow.

-Input
The input includes a PROCESS sequence in the following format: 
process_1: <process_1> {DELIMITER} process_2: <process_2> {DELIMITER} ... process_k: <process_k>

For each process in the sequence, the detailed information is also given, including the entity of the process corresponding to a SYSTEM MODULE or an OBJECT, the action taken in the process and the status. The detailed process information is in the following format: 
process 1: <process_1> {DELIMITER} <entity_num_process_1> entity(s): <entities_process_1> {DELIMITER} <action_num_process_1>  action(s): <actions_process_1> {DELIMITER} <statuses_process_1> status(s): <status_process_1>
process 2: <process_2> {DELIMITER} <entity_num_process_2> entity(s): <entities_process_2> {DELIMITER} <action_num_process_2> action(s): <actions_process_2> {DELIMITER} <statuses_process_2> status(s): <status_process_2>
...
process k: <process_k> {DELIMITER} <entity_num_process_k> entity(s): <entities_process_k> {DELIMITER} <action_num_process_k> action(s): <actions_process_k> {DELIMITER} <statuses_process_k> status(s): <status_process_k>

-Output-
Generate the summary of the sequential execution represented by the sequence.  Make sure to include information collected from all the processes. IMPORTANT: Desribe the execution sequence without using the word "process". 

######################
-Real Data-
######################
-Input-
PROCESS sequence: {input_process_seq}
PROCESS details: 
{input_process_desc}
######################
Output:"""

PROCESS_SEQ_SUMMARY_PROMPT_V2 = """
-Role-
You are an expert in understanding and summarizing the sequential workflow of system execution. 

-Goal- 
Given a sequence of processes  which represents an execution workflow. Please summarize the sequence into a single, comprehensive description in respect to the execution flow.

-Input
The input includes a PROCESS sequence in the following format: 
1: <process_1> {DELIMITER} 2: <process_2> {DELIMITER} ... k: <process_k>

-Output-
Generate the summary of the sequential execution represented by the sequence.  Make sure to include information collected from all the processes. IMPORTANT: Desribe the execution sequence without using the word "process". 

######################
-Real Data-
######################
-Input-
PROCESS sequence: {input_process_seq}
######################
Output:"""


PROCESS_SEQ_DETECT_PROMPT = """
-Role-
You are an expert in understanding normal_flows execution workflows as well as detecting abnormal_flows execution workflows. 

-Goal- 
Given several example sequences of PROCESSES which represents normal_flows execution workflows as the context, decide if a test PROCESS sequence indicates an abnormal_flows execution workflow. 

-Context-
The context includes k (k = {example_num}) examples of PROCESS sequences as normal_flows execution workflows. Each example includes the actual PROCESS sequence and a text description in the following format:
Example sequence i - sequence:  <process_1> {DELIMITER}  <process_2> {DELIMITER} ...  <process_n>
Example sequence i - description: <example_sequence_desc>

-Input-
The input includes a test PROCESS sequence and its description in the following format: 
Test sequence - sequence:  <process_1> {DELIMITER} <process_2> {DELIMITER} ...  <process_n>
Test sequence - description: <test_sequence_desc>

-Output-
Summarize the test sequence, and return the prediction and reason of the prediction for the test sequence
- prediction: Return "Normal" or "Abnormal" as the decision for the test PROCESS sequence.
- reason: Return the reason for the prediction
Generate the output in the following format:
{
"prediction": <prediction>,
"reason": <reason>
}

######################
-Real Data-
######################
-Context-
{example_sequence_context}

-Input-
Test sequence  - sequence: {input_process_seq}
Test sequence  - description: {input_process_desc}
######################
Output:"""


PROCESS_SEQ_DETECT_PROMPT_WITHOUT_SUMMARIES = """
-Role-
You are an expert in understanding normal_flows execution workflows as well as detecting abnormal_flows execution workflows. 

-Goal- 
Given several example sequences of PROCESSES which represents normal_flows execution workflows as the context, decide if a test PROCESS sequence indicates an abnormal_flows execution workflow. 

-Context-
The context includes k (k = {example_num}) examples of PROCESS sequences as normal_flows execution workflows. Each example includes the actual PROCESS sequence in the following format:
Example sequence i - sequence: 1: <process_1> {DELIMITER} 2: <process_2> {DELIMITER} ...  n: <process_n>

-Input-
The input includes a test PROCESS sequence in the following format: 
Test sequence - sequence: 1: <process_1> {DELIMITER} 2: <process_2> {DELIMITER} ... n: <process_n>

-Output-
Understand and examine the example sequences carefully, and return the prediction and reason of the prediction for the test sequence
- prediction: Return "Normal" or "Abnormal" as the decision for the test PROCESS sequence.
- reason: Return the reason for the prediction
Generate the output in the following format:
{
"prediction": <prediction>,
"reason": <reason>
}
-IMPORTRANT-
PLEASE REVIEW THE EXAMPLE AND TEST SEQUENCES CAREFULLY AND GIVE CONCRETE REASONS INSTEAD VAGUE ONES FOR YOUR DECISION!

######################
-Real Data-
######################
-Context-
{example_sequence_context}

-Input-
Test sequence  - sequence: {input_process_seq}
######################
Output:"""


PROCESS_SEQ_VERIFY_AND_MASK_PROMPT = """
-Role-
You are an expert in understanding normal_flows execution workflows as well as detecting abnormal_flows execution workflows. 

-Input-
The input includes a test PROCESS sequence and its description in the following format: 
process_1: <process_1>
process_2: <process_2>
 ... 
process_n: <process_n>

-Steps- 
1. Decide if a test PROCESS sequence indicates an abnormal_flows execution workflow. 
2. If it is decided as abnormal_flows, find the most important keywords lead to the decision
3. Replace the keywords in the sequence with words that lead to a normal_flows decision
4. Verify if the modified sequence indicates a normal_flows execution, if yes, generate the following output, if no, repeat the 2-3 steps.

-Output-
Output the original  in the following json format:
{
"prediction": <"Normal" or "Abnormal" as the decision for the test PROCESS sequence>,
"abnormal_keywords": <list of the abnormal_flows keywords, delimited by comma. make it "none" if the prediction is "Normal">,
"modified_sequence": <modified sequence, make it the same with the original sequence if the prediction is "Normal">,
"modified_process_IDs": <the IDs of the modified processes in the modified sequence, delimited by comma. make it "none" if the prediction is "Normal">
}

######################
-Real Data-
######################
-Input-
Test sequence  - sequence: {input_process_seq}
######################
Output:"""


PROCESS_VERIFY_AND_MASK_PROMPT = """
-Role-
You are an expert in understanding normal_flows execution workflows as well as detecting abnormal_flows execution processes. 

-Input-
The input includes an execution process

-Steps- 
1. Decide if a test process indicates an abnormal_flows execution workflow. 
2. If it is decided as abnormal_flows, find the most important keywords lead to the decision
3. Replace the keywords in the process with words that lead to a normal_flows decision
4. Verify if the modified process indicates a normal_flows execution, if yes, generate the following output, if no, repeat the 2-3 steps.

-Output-
Output the original  in the following json format:
{
"prediction": <"Normal" or "Abnormal" as the decision for the test process>,
"abnormal_keywords": <list of the abnormal_flows keywords, delimited by comma. make it "none" if the prediction is "Normal">,
"modified_process": <modified process which is normal_flows, make it the same with the original process if the prediction is "Normal">
}

######################
-Real Data-
######################
-Input-
{input_process_seq}
######################
Output:"""

PROCESS_DETECT_PROMPT = """
-Role-
You are an expert in understanding normal execution processes as well as detecting abnormal execution processes. 

-Input-
The input includes a log message representing an execution process

-Steps- 
1. Decide if a log message indicates an abnormal execution process. Return normal if you are not sure if it is abnormal or not.

-Output-
Output the result in the following json format:
{
"prediction": <"Normal" or "Abnormal">,
"reason": <the reason for the prediction>,
}

######################
-Real Data-
######################
-Input-
{input_process}
######################
Output:"""