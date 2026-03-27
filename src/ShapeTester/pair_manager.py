from enum import Enum
from inputSpace_manager import InputSpaceManager
from utils.log_manager import setup_logger

logger = setup_logger(sub_dir="core")

AssignedParam = tuple[int,int]
AssignedParamTuple = tuple[AssignedParam, AssignedParam]
testSegment = tuple[int,...]
testcase = list[int]

class CoverStatus(Enum):
    """Coverage status for a parameter pair."""
    Invalid = -1
    COVERED = 1
    NOT_COVERED = 0

class pairwise_tuple:
    def __init__(self, param_tuple:AssignedParamTuple):
        self.param_tuple:AssignedParamTuple = param_tuple
        self.status:CoverStatus = CoverStatus.NOT_COVERED
        self.related_segments:set[Segment] = set()
        self.value:float = 0
        self.failed_cost_score = 0
    def __hash__(self):
        return hash(self.param_tuple)
        
    def add_related_segment(self, seg:"Segment"):
        self.related_segments.add(seg)
    
    def remove_related_segment(self, seg:"Segment"):
        self.related_segments.remove(seg)
    
    def calculate_value(self):
        self.value = 1 + 1/(len(self.related_segments) + self.failed_cost_score)
        
class Segment:
    def __init__(self, segment:testSegment):
        self.segment = segment
        self.Cover_Tuple:set[pairwise_tuple] = set()
    
    def add_cover_tuple(self, tuple:pairwise_tuple):
        self.Cover_Tuple.add(tuple)
    
    def remove_cover_tuple(self, tuple:pairwise_tuple):
        self.Cover_Tuple.remove(tuple)
    
    def __hash__(self) -> int:
        return hash(self.segment)
        
class Pairwise_Tuple_Manager:
    def __init__(self, input_space_manager:InputSpaceManager):
        self.param_num = input_space_manager.param_num
        all_pairs:list[AssignedParamTuple] = input_space_manager.all_param_pairs()
        self.pair_map = {pair:pairwise_tuple(pair) for pair in all_pairs}
        all_segments:list[testSegment] = input_space_manager.all_segments()
        self.segment_map = {seg:Segment(seg) for seg in all_segments}
        self._post_init()
        logger.info("Pairwise_Tuple_Manager initialized: pairs=%d segments=%d", len(self.pair_map), len(self.segment_map))
    
    def _post_init(self):
        for segment_id,seg in self.segment_map.items():
            for i in range(self.param_num):
                for j in range(i+1,self.param_num):
                    covered_tuple = ((i,segment_id[i]), (j,segment_id[j]))
                    seg.add_cover_tuple(self.pair_map[covered_tuple])
                    self.pair_map[covered_tuple].add_related_segment(seg)
        for pairs in self.pair_map.values():
            pairs.calculate_value()
        logger.debug("Post init complete for pairwise tuples.")
    
    def complete_one_segment(self, segment:testSegment):
        seg_obj = self.segment_map.pop(segment)
        covered_pair = []
        for pair in seg_obj.Cover_Tuple:
            covered_pair.append(self.pair_map.pop(pair.param_tuple))
        for pair in covered_pair:
            for seg in pair.related_segments:
                seg.remove_cover_tuple(pair)
                if not seg.Cover_Tuple and seg.segment in self.segment_map:
                    self.segment_map.pop(seg.segment)
        logger.debug("Completed segment: %s", segment)
                                                            
    def ban_one_segment(self, segment:testSegment):
        seg_obj = self.segment_map.pop(segment)

        for pair,pair_obj in list(self.pair_map.items()):
            first_param, second_param = pair
            if segment[first_param[0]] == first_param[1]:
                pair_obj.failed_cost_score += 1
            if segment[second_param[0]] == second_param[1]:
                pair_obj.failed_cost_score += 1
                
        for pair in seg_obj.Cover_Tuple:
            pair.remove_related_segment(seg_obj)
            if not pair.related_segments:
                self.pair_map.pop(pair.param_tuple)
            else:
                pair.calculate_value()
        logger.debug("Banned segment: %s", segment)
                
    def ban_one_comb(self, comb:tuple[tuple[int,int],...]):
        invalid_seg = []
        for seg_id, seg in self.segment_map.items():
            if all([seg_id[i] == v for i,v in comb]):
                invalid_seg.append(seg_id)
        for seg in invalid_seg:
            self.ban_one_segment(seg)
        logger.info("Banned combination %s, removed %d segments", comb, len(invalid_seg))
                
    def select_best_segment(self):
        best_score = -1
        best_seg = None
        for seg_id, seg in self.segment_map.items():
            seg_score = 1
            for pair in seg.Cover_Tuple:
                seg_score += pair.value
            if seg_score > best_score:
                best_score = seg_score
                best_seg = seg_id
        logger.debug("Selected best segment: %s", best_seg)
        return best_seg
        
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
    pm = Pairwise_Tuple_Manager(input_space_manager)
    print(len(pm.pair_map),len(pm.segment_map))
    pm.ban_one_comb(((1,0),))
    print(len(pm.pair_map),len(pm.segment_map))