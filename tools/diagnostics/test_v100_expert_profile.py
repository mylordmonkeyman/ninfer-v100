import unittest
from v100_expert_profile import build_profile, evaluate


class ExpertProfileTest(unittest.TestCase):
    def test_ranking_and_held_out_hits(self):
        train = [[0] * 512 for _ in range(48)]
        train[0][9], train[0][2], train[0][4] = 8, 8, 3
        profile = build_profile(train, 'a' * 64, 'decode', 'model', 'weights')
        self.assertEqual(profile['ranking'][0][:3], [2, 9, 4])
        held_out = [[0] * 512 for _ in range(48)]
        held_out[0][2], held_out[0][4] = 3, 7
        held_out[1][2] = 5
        report = evaluate(profile, held_out, [2] + [0] * 47)
        self.assertEqual(report['projected_hits'], 3)
        self.assertEqual(report['routes'], 15)
        self.assertEqual(report['hit_fraction'], 0.2)
        self.assertFalse(report['qualified'])

    def test_reject_bad_geometry_and_artifact(self):
        rows = [[0] * 512 for _ in range(48)]
        with self.assertRaises(ValueError):
            build_profile(rows, 'model-name', 'decode', 'model', 'weights')
        profile = build_profile(rows, 'a' * 64, 'decode', 'model', 'weights')
        for slots in ([1] * 47, [513] * 48, [True] * 48):
            with self.assertRaises(ValueError):
                evaluate(profile, rows, slots)


if __name__ == '__main__':
    unittest.main()
