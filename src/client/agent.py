from typing import List


class AgentClient:
    def __init__(self, *args, **kwargs):
        self.shared_memory = None  # 初始化为 None

    def set_shared_memory(self, shared_memory):
        self.shared_memory = shared_memory  # 设置共享记忆

    def perform_task(self, task_data):
        if self.shared_memory is None:
            return self.inference(None, task_data)

        # 从共享记忆中获取历史数据
        history = self.shared_memory.retrieve(self.__class__.__name__)

        # 结合历史数据和当前任务数据进行推理
        task_result = self.inference(history, task_data)

        # 更新共享记忆
        self.shared_memory.store(self.__class__.__name__, task_result)

        return task_result

    def inference(self, history, task_data):

        raise NotImplementedError()



    @staticmethod
    def create_with_shared_memory(agent_config, shared_memory):
        # 创建代理实例
        agent_instance = agent_config.create()
        # 设置共享记忆
        agent_instance.set_shared_memory(shared_memory)
        return agent_instance