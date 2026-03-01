
class TimeTracker:
    def __init__(self):
        self.sequence_num = []
        self.test_sequence_breakdown = 0
        self.test_sequence_breakdown_counter = 0

        self.pattern_test = 0
        self.pattern_test_count = 0

        self.knowledge_test = 0
        self.knowledge_test_count = 0

    def update_sequence_num(self, seq_num):
        self.sequence_num.append(seq_num)

    def update_sequence_breakdown(self, time):
        self.test_sequence_breakdown+=time
        self.test_sequence_breakdown_counter +=1

    def update_pattern_test(self, time):
        self.pattern_test+=time
        self.pattern_test_count+=1

    def update_knowledge_test(self, time):
        self.knowledge_test+=time
        self.knowledge_test_count+=1

    def report(self):
        print("=== Test Time Tracker ===")
        print("Sequence num:  {}".format(self.sequence_num))
        print("Sequence breakdown: {}, counter:{}".format(self.test_sequence_breakdown, self.test_sequence_breakdown_counter))
        print("Pattern Matching Test: {}, counter {}".format(self.pattern_test, self.pattern_test_count))
        print("Knowledge Test: {}, counter:{}".format(self.knowledge_test, self.knowledge_test_count))