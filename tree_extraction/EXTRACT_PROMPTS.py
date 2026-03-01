# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to a system process, identify the relevant entities, action and the status of the process.

-Steps-
1. source_entity: Find the name of the source entity that initiates the process, make sure it is a legit word with actual meaning, make it none if not found
2. source_entity_type: Find the entity type of the source entity found in step 1, it should be one of the following types: [{entity_types}], make it none if the source entity is not found
3. target_entity: Find the name of the target entity that undertakes the process, make sure it is a legit word with actual meaning, make it none if not found
4. target_entity_type: Find the entity type of the target entity found in step 3, it should be one of the following types: [{entity_types}], make it none if the target entity is not found
5. action: Find the action of the process that the source takes on the target entity, make it none if not found
6. status: Find the status of the process, usually as an adjective describing the process, or maybe a noun as error, exception or failure, etc. make it none if not found
7. summary: summarize the text document as a process in approximately 5 words
8. Output the above steps in a json format as follows:
{"source_entity": <source_entity>,
"source_entity_type":<source_entity_type>,
"target_entity": <target_entity>,
"target_entity_type": <target_entity_type>,
"action": <action>,
"status": <status>,
"summary": <summary>
}

######################
-Examples-
######################
Example 1: 
Entity_types: OBJECT
Text: Receiving block <*> src: <*> dest: <*>
######################
Output:
{
"source_entity": "none",
"source_entity_type": "none",
"target_entity": "block",
"target_entity_type": "OBJECT",
"action": "receiving",
"status": "none",
"summary": "receiving block"
}
######################

Example 2:
Entity_types: SYSTEM MODULE, OBJECT
Text: PacketResponder <*> for block <*> terminating
######################
Output:
{
"source_entity": "PacketResponder",
"source_entity_type": "SYSTEM MODULE",
"target_entity": "block",
"target_entity_type": "OBJECT",
"action": "terminating",
"status": "none",
"summary": "PacketResponder terminating"
}

######################
Example 3:
Entity_types: "SYSTEM MODULE", "OBJECT"
Text: Verification succeeded for <*>
######################
Output:
{
"source_entity": "none",
"source_entity_type": "none",
"target_entity": "none",
"target_entity_type": "none",
"action": "verify",
"status": "successful",
"summary": "Successful verification"
}

######################
Example 4:
Entity_types: "SYSTEM MODULE",  "OBJECT"
Text: PacketResponder <*> <*> Exception java.nio.channels.ClosedByInterruptException
######################
Output:
{
"source_entity": "PacketResponder",
"source_entity_type": "SYSTEM MODULE",
"target_entity": "none",
"target_entity_type": "none",
"action": "none",
"status": "exception",
"summary": "PacketResponder with ClosedByInterruptException java exception"
}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

REPEAT_PROMPT = "The returned output is not in a correct json format. Please regenerate it in the DEFINED json format. "


ENTITY_FILLING_PROMPT = """
-Goal-
Given a text document that is potentially relevant to a system process, try to select the correct entity of the process from a given entity list. 

-Steps-
1. Select the word of the MOST important entity from the entity word list which initiates or undertakes the process.
2. If the correct option is not in the list, generate a word of MOST important entity.
3. If the entity involves multiple words, concatenate them in camel case.
4. Find the entity type of the entity, it should be in one of the following types {entity_types}
5. Indicate the source of the entity word, should be either "selected" or "generated".

Return the output in the following json format:
{"entity": <entity_word>,
"entity_type": <entity_type>,
"source": "selected" or "generated"}


-Example 1-
Entity_list: "writeBlock",  "NameSystem", "PacketResponder"
Text: writeBlock received exception 
Output: 
{
"entity": "writeBlock",
"entity_type": "SYSTEM MODULE",
"source": "selected"
}

######################
-Real Data-
######################
Entity_list: {entity_list}
Text: {input_text}
######################
Output:"""


ACTION_FILLING_PROMPT = """
-Goal-
Given a text document that is potentially relevant to a system process, try to select the correct ACTION word related to the given ENTITY from a given action word list. 

-Steps-
1. Select the ACTION word from the ACTION word list which describes the action related to the given ENTITY in the process. Focus only on the given entity and ignore irrelevant action words.
2. If there is no correct option in the ACTION list, generate the best word which best describing the action related to the given ENTITY in the process. Focus only on the given entity and ignore irrelevant action words.
3. If the ACTION word involves multiple words, concatenate them in camel case.
4. Indicate the source of the ACTION word, should be either "selected" or "generated".

Return the output in the following json format:
{"action": <action_word>,
"source": "selected" or "generated"}


-Example 1-
Text: Exception in receiveBlock for block  
Entity: "block"
Action_list: "write",  "delete", "receive"
Output: 
{
"action": "receive",
"source": "selected"
}

######################
-Real Data-
######################
Text: {input_text}
Entity: {entity}
Action_list: {action_list}
######################
Output:"""
