from typing import List


class AgentClient:
    def __init__(self, *args, **kwargs):
        self.shared_memory = None  # Initialized to None

    def set_shared_memory(self, shared_memory):
        self.shared_memory = shared_memory  # Set up shared memory

    def perform_task(self, task_data):
        if self.shared_memory is None:
            return self.inference(None, task_data)

        # Get historical data from shared memory
        history = self.shared_memory.retrieve(self.__class__.__name__)

        # Combine historical data and current task data for inference
        task_result = self.inference(history, task_data)

        # Update shared memory
        self.shared_memory.store(self.__class__.__name__, task_result)

        return task_result

    def inference(self, history, task_data):

        raise NotImplementedError()



    @staticmethod
    def create_with_shared_memory(agent_config, shared_memory):
        # Create proxy instance
        agent_instance = agent_config.create()
        # Set up shared memory
        agent_instance.set_shared_memory(shared_memory)
        return agent_instance
