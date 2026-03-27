
from dataclasses import fields
from itertools import product
from utils.log_manager import setup_logger

logger = setup_logger(sub_dir="core")

class InputSpaceManager:
    def __init__(self, input_space):
        self.input_space = input_space
        self.param_num = len(fields(input_space))
        self.param2int = {field.name: i for i, field in enumerate(fields(input_space))}
        self.int2param = {i: field_name for field_name,i in self.param2int.items()}
        self.int2range = {self.param2int[field.name]: getattr(input_space, field.name) for field in fields(input_space)}
        self.int2length = {i:len(r) for i,r in self.int2range.items()}
        logger.info("InputSpaceManager initialized with %d params.", self.param_num)
        
    def all_param_pairs(self):
        all_param_pairs = []
        for i in range(self.param_num):
            for j in range(i+1,self.param_num):
                    for param_i in range(self.int2length[i]):
                        for param_j in range(self.int2length[j]):
                            all_param_pairs.append(((i, param_i),(j, param_j)))
        logger.debug("Generated %d param pairs.", len(all_param_pairs))
        return all_param_pairs
    
    def all_segments(self):
        all_param_range = [[j for j in range(self.int2length[i])] for i in range(self.param_num)]
        segments = list(product(*all_param_range))
        logger.debug("Generated %d segments.", len(segments))
        return segments
    
    def segment_to_testcase(self, segment:tuple[int,...]) -> dict:
        test_case = {}
        for i, param_value_index in enumerate(segment):
            field_name = self.int2param[i]
            test_case[field_name] = self.int2range[i][param_value_index]
        logger.debug("Segment to testcase: %s", segment)
        return test_case
    
    def testcase_to_segment(self, testcase:dict) -> tuple[int,...]:
        segment = []
        for i in range(self.param_num):
            field_name = self.int2param[i]
            param_value = testcase.get(field_name,None)
            param_value_index = self.int2range[i].index(param_value)
            segment.append(param_value_index)
        logger.debug("Testcase to segment: %s", segment)
        return tuple(segment)

if __name__ == "__main__":
    from dataclasses import dataclass, field
    @dataclass
    class InputSpace:
        # output_mode: 离散型参数，所有可能取值
        output_mode: list[str] = field(default_factory=lambda: ["int", "one_hot", "multi_hot", "count", "tf_idf"])
        # num_bins: 连续型整数参数，离散化为包含边界值和5个典型值的列表
        num_bins: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 100])
        # bin_boundaries: 连续型列表参数，选取6个典型取值（包括None）以覆盖常见场景
        bin_boundaries: list[list[float]|None] = field(default_factory=lambda: [
            None,
            [0.0],
            [0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [-1.0, 0.0, 1.0]
        ])
    input_space = InputSpace()
    input_space_manager = InputSpaceManager(input_space)
    print(len(input_space_manager.all_param_pairs()))
    print(input_space_manager.int2length)
    print(len(input_space_manager.all_segments()))