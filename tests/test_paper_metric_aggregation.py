"""Independent frozen-metric oracle for exact confusion recovery and IoU definitions."""
import ast
import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_paper_recipe_study as report


class MetricAggregationTest(unittest.TestCase):
    def rows(self,counts):
        path=ROOT/'experiments/exp00_baseline/kvasir_test.py'
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_batch_metrics')
        namespace=dict(torch=torch)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        pixels=352*352;prediction=torch.zeros(len(counts),1,pixels);target=torch.zeros_like(prediction)
        for i,(tp,fp,fn,tn) in enumerate(counts):
            self.assertEqual(tp+fp+fn+tn,pixels)
            prediction[i,0,:tp+fp]=1
            target[i,0,:tp]=1;target[i,0,tp+fp:tp+fp+fn]=1
        metrics=namespace['compute_batch_metrics'](prediction,target)
        return [dict(id=str(i),foreground_pixels=int(target[i].sum()),**{k:v[i] for k,v in metrics.items()}) for i in range(len(counts))]

    def test_recovers_counts_from_frozen_float32_oracle(self):
        n=352*352
        counts=[(0,0,0,n),(0,500,0,n-500),(0,0,123,n-123),(1,0,1,n-2),
            (12345,6789,2345,n-21479),(n-3,1,1,1)]
        rows=self.rows(counts);result=report.aggregation_metrics(rows)
        expected={key:sum(c[i] for c in counts) for i,key in enumerate(('tp','fp','fn','tn'))}
        self.assertEqual(result['confusion_counts'],expected)
        self.assertEqual(result['images'],len(counts))

    def test_global_and_image_means_are_distinct(self):
        n=352*352;counts=[(100,0,0,n-100),(10000,10000,0,n-20000)]
        rows=self.rows(counts);result=report.aggregation_metrics(rows)
        foreground=10100/20100;background=(2*n-20100)/(2*n-10100)
        self.assertAlmostEqual(result['foreground_global_iou_percent'],100*foreground)
        self.assertAlmostEqual(result['global_class_mean_iou_percent'],50*(foreground+background))
        self.assertNotAlmostEqual(result['foreground_image_mean_iou_percent'],result['foreground_global_iou_percent'])

    def test_cross_checks_saved_metrics_and_handles_undefined_class(self):
        n=352*352;rows=self.rows([(100,20,30,n-150)])
        corrupted=copy.deepcopy(rows);corrupted[0]['precision']=.2
        with self.assertRaisesRegex(ValueError,'disagree'):
            report.aggregation_metrics(corrupted)
        empty=report.aggregation_metrics(self.rows([(0,0,0,n)]))
        self.assertIsNone(empty['foreground_global_iou_percent'])
        self.assertEqual(empty['global_class_mean_iou_percent'],100.)


if __name__=='__main__':
    unittest.main()
