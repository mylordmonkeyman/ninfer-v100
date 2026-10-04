import json
from pathlib import Path
import tempfile
import unittest

from v100_sv1_residency import compare_logits, parse_run, parse_oracle


class ResidencyReportTest(unittest.TestCase):
    def test_full_oracle_failure_remains_visible_and_early_exit_rejected(self):
        metrics = dict(positions=4096, nonfinite_positions=0, mean_kl=.1,
                       p99_kl=1.8, top1_agreement=.91, relative_mean_nll_delta=.01,
                       maximum_logit_error=2)
        metrics['host_expert.completed_layer_calls'] = 4096*48
        metrics['host_expert.expert_pairs'] = 4096*48*10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'oracle.log'
            content = '\n'.join(f'phase11.{key}={value}' for key,value in metrics.items())
            path.write_text(content+'\nFAIL: Phase 11 teacher-forced oracle gate\n')
            self.assertFalse(parse_oracle(path,1)['independent_oracle_passed'])
            with self.assertRaises(ValueError): parse_oracle(path,0)
            path.write_text(content+'\nFAIL: Phase 11 vertical slice: incomplete\n')
            with self.assertRaises(ValueError): parse_oracle(path,1)
            path.write_text(content.replace('phase11.positions=4096','phase11.positions=64')+
                            '\nFAIL: Phase 11 teacher-forced oracle gate\n')
            with self.assertRaises(ValueError): parse_oracle(path,1)

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
