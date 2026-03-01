from executor.executor import *

# local detection configuration
configs = {'dataset': "BGL",
           "entity_level": True,
           "action_level": True,
           "status_level": True,
           "lazy_detect": True,  # if disabled, krone will still execute even an abnormal krone-seq is detected
           "hardcode_kleene_pattern_summary":True,
           "train_percent": 80,
           "detect_mode": "local",
           "automaton_adjustment": True,  # adjust for sliding windows
           "edge_consecutive_sensitive": False, # adjust for sliding windows

           #  LLM related settings, ignored when local detection
           "test_1_percent": 10,  # enable llm for 20% of the test, ignored when local detection
           "k_neighbors": 5,  # number of demonstration examples, ignored when local detection

           # krone-seq knowledge base, caching and reusing
           "load_history_test_embedding": False, # load the cached embedding of krone-seqs of the test set, stored in output/{dataset}/test_emebdding_all.csv
           "load_history_test_summary": False, # load the cached summary of krone-seqs of the test set, stored in output/{dataset}/test_summary_all.csv
           "load_history_test_knowledge": False, # load the cached knowledge (LLM detection result) of krone-seqs of the test set, stored in output/{dataset}/test_knowledge_all.csv
           "store_test_knowledge_as_history": False, # if cache the test knowledge (LLM detection result) of krone-seqs
           }

# LLM detection configuration
# configs = {'dataset': "BGL",
#            "entity_level": True,
#            "action_level": True,
#            "status_level": True,
#            "lazy_detect": True,
#            "hardcode_kleene_pattern_summary":True,
#            "train_percent": 20,
#            "test_1_percent": 20,
#            "detect_mode": "local",
#            "automaton_adjustment": True,
#            "edge_consecutive_sensitive": False,
#            "k_neighbors": 5,
#            "load_history_test_embedding": True,
#            "load_history_test_summary": True,
#            "load_history_test_knowledge": True,
#            "store_test_knowledge_as_history": True,
#            }

executor = Executor()
executor.load_configs(configs)
sequence_path = f"../data/{executor.dataset}/{executor.dataset}_demo_sequences.csv"
sequence_df = pd.read_csv(sequence_path)
executor.build(sequence_df)
executor.run()
