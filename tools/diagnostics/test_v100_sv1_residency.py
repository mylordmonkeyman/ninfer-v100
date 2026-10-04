import json
from pathlib import Path
import tempfile
import unittest

from v100_sv1_residency import compare_logits, parse_run


class ResidencyReportTest(unittest.TestCase):
    def test_incomplete_and_corrupt_measurements_rejected(self):
        rows = [dict(sv1='startup'), dict(sv1='complete',tokens=6,decode_top1=[0]*4)]
        rows += [dict(sv1='measurement',shape=shape,tokens=count,seconds=1,
                      tokens_per_s=count,maximum_outstanding=4)
                 for shape,count in [('prefill',2),('decode',4)]]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run.log'
            def write(records,marker=True):
                path.write_text('\n'.join(json.dumps(r) for r in records)+'\n'+
                    ('PASS: SV1 real-artifact workload finite and committed' if marker else ''))
            write(rows)
            self.assertEqual(parse_run(path)['complete']['tokens'],6)
            for bad in (rows[:-1],rows+[rows[-1]]):
                write(bad)
                with self.assertRaises(ValueError): parse_run(path)
            write(rows,False)
            with self.assertRaises(ValueError): parse_run(path)
            rows[-1]['tokens_per_s']=1
            write(rows)
            with self.assertRaises(ValueError): parse_run(path)

    def test_pairwise_numerical_screen(self):
        same=compare_logits([0.,1.,2.],[0.,1.,2.])
        self.assertTrue(same['bitwise_equal'])
        self.assertEqual(same['final_position_kl'],0)
        different=compare_logits([0.,1.,2.],[2.,1.,0.])
        self.assertFalse(different['bitwise_equal'])
        self.assertGreater(different['final_position_kl'],0)
        self.assertEqual(different['final_position_maximum_error'],2)
        with self.assertRaises(ValueError): compare_logits([0.,1.],[0.])


if __name__ == '__main__':
    unittest.main()
