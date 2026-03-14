class FullyMixedTopology:
    def set_initial_opinion(self, opinions):
        self.opinions = list(opinions)

    def get_opinions(self):
        return list(self.opinions)

    def get_neighbours(self, node_id):
        # Remove the element at the chosen index
        opinions_copy = list(self.opinions)
        opinions_copy.pop(node_id)
        return opinions_copy

    def change_one_opinion(self, agent_id_1, new_opinion):
        opinions_agent1_change = list(self.opinions)  # first make a copy of the distribution
        opinions_agent1_change[agent_id_1] = new_opinion
        return opinions_agent1_change
