import json
import unittest
from v100_sv3_compare import summarize


class ComparisonEvidence(unittest.TestCase):
    def fixture(self):
        return [dict(layer=layer,sample=sample,tokens=1227,nonfinite=0,
                     isolated_replay_differences=0,avx2_available=True,
                     cpu_reference_nrmse=1e-4,cpu_reference_cosine=1.0,
                     maximum_error=1e-3,avx2_reference_nrmse=2e-5,gpu_avx2_nrmse=1e-4)
                for layer in range(48) for sample in range(8)]

    def text(self, rows):
        return '\n'.join('v100.stream_compare='+json.dumps(r) for r in rows)

    def test_complete_evidence_reports_profile_failure_without_promoting(self):
        rows=self.fixture(); rows[0]['cpu_reference_nrmse']=.003
        report=summarize(self.text(rows))
        self.assertFalse(report['qualified'])
        self.assertEqual(report['cpu_profile_criterion_failures'],1)
        self.assertTrue(report['isolated_gpu_replay_exact'])

    def test_incomplete_or_duplicate_evidence_rejected(self):
        rows=self.fixture()
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): summarize(self.text(bad))

    def test_nonfinite_or_changed_gpu_replay_rejected(self):
        for key,value in [('nonfinite',1),('cpu_reference_nrmse',float('nan')),
                          ('isolated_replay_differences',1)]:
            rows=self.fixture(); rows[0][key]=value
            with self.assertRaises(ValueError): summarize(self.text(rows))


if __name__=='__main__': unittest.main()
