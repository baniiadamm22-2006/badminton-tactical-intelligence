import unittest
from src.rally_candidates import scene_intervals, smooth_boolean, propose, evaluate

class TestRallyCandidates(unittest.TestCase):
    def test_scenes(self):
        self.assertEqual(scene_intervals(100, [20, 60, 20]), [(0, 19), (20, 59), (60, 99)])

    def test_fill_short_gaps_not_edges(self):
        self.assertEqual(smooth_boolean([False, True, False, True, False], 1),
                         [False, True, True, True, False])

    def test_cut_never_crossed(self):
        tracks = {i: {'near': {1}, 'far': {2}} for i in range(100)}
        candidates = propose(100, [50], tracks, 10, min_seconds=1)
        self.assertEqual([(x['start_frame'], x['end_frame']) for x in candidates], [(0,49),(50,99)])

    def test_no_detection_no_rally(self):
        self.assertEqual(propose(100, [], {}, 10), [])

    def test_evaluation(self):
        props = [{'proposal_id': 1, 'start_frame': 10, 'end_frame': 30}]
        refs = [{'rally_id': 1, 'start_frame': 12, 'end_frame': 28, 'annotation_status': 'verified'}]
        self.assertEqual(evaluate(props, refs)['matched_at_iou_0_5'], 1)

if __name__ == '__main__': unittest.main()
