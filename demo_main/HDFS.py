from executor.executor import *

if __name__ == '__main__':
    configs = {'dataset': "HDFS",
               "entity_level": True,
               "action_level": True,
               "status_level": True,
               "lazy_detect": True,
               "train_percent": 80,
               "test_1_percent": 10,  # enable llm for 20% of the test
               "detect_mode": "local",
               "automaton_adjustment": False,  # adjust for sliding windows
               "edge_consecutive_sensitive": False,  # adjust for sliding windows
               "k_neighbors": 5,
               "load_history_test_embedding": False,
               "load_history_test_summary": False,
               "load_history_test_knowledge": False,
               "store_test_knowledge_as_history": False,

               }

    # configs = {'dataset': "HDFS",
    #            "entity_level": True,
    #            "action_level": True,
    #            "status_level": True,
    #            "lazy_detect": True,
    #            "train_percent": 80,
    #            "test_1_percent": 10,  # enable llm for 20% of the test
    #            "detect_mode": "local",
    #            "automaton_adjustment": False,  # adjust for sliding windows
    #            "edge_consecutive_sensitive": False,  # adjust for sliding windows
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
