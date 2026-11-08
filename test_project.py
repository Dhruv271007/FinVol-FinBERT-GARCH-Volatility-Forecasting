"""Behavioral checks for forecast timing, recursion, constraints, and losses."""
import importlib
import unittest
import numpy as np
import pandas as pd
from corrected_data import issuance_times, aggregate_news
from volatility_models import variance_path, fit_model, qlike_loss, diebold_mariano, holm_adjust


class ProjectTests(unittest.TestCase):
    def test_recursion_matches_hand_calculation(self):
        r=np.array([1.,-2.,3.]); s=np.array([0.,-.8,.4])
        h=variance_path([0,.6,.1,.7,.5],r,s,2.)
        np.testing.assert_allclose(h,[2.,1.7,2.39])

    def test_future_returns_do_not_change_earlier_forecasts(self):
        r=np.linspace(-2,2,50); p=[0,.2,.1,.8]
        original=variance_path(p,r,initial_variance=2)
        changed=r.copy(); changed[30:]+=10
        altered=variance_path(p,changed,initial_variance=2)
        np.testing.assert_array_equal(original[:31],altered[:31])
        self.assertNotEqual(original[31],altered[31])

    def test_calendar_early_close_and_dst(self):
        issues=issuance_times(pd.to_datetime(["2025-11-28","2025-12-01"]))
        self.assertEqual(str(issues[0]),"2025-11-28 18:05:00+00:00")
        self.assertEqual(str(issues[1]),"2025-12-01 21:05:00+00:00")
        summer=issuance_times(pd.to_datetime(["2025-07-07"]))
        self.assertEqual(str(summer[0]),"2025-07-07 20:05:00+00:00")

    def test_weekend_and_cutoff_assignment(self):
        issues=issuance_times(pd.to_datetime(["2025-03-06","2025-03-07","2025-03-10","2025-03-11"]))
        news=pd.DataFrame({"date":pd.to_datetime(["2025-03-07 21:05:00Z","2025-03-08 12:00:00Z","2025-03-10 20:06:00Z"]),
                           "sentiment":[.2,.8,-.5]})
        means,counts=aggregate_news(news,issues)
        np.testing.assert_array_equal(counts,[0,1,1,1])
        np.testing.assert_allclose(means,[0,.2,.8,-.5])

    def test_constraints_hold_for_extreme_future_sentiment(self):
        rng=np.random.default_rng(42)
        r=rng.normal(0,1.5,400); s=rng.uniform(-1,1,400)
        fit=fit_model(r,s); p=fit.x
        self.assertLessEqual(p[2]+p[3],.999+1e-8)
        self.assertGreaterEqual(p[1]-abs(p[4]),1e-6-1e-9)
        h=variance_path(p,np.zeros(100),np.tile([-1.,1.],50),fit.initial_variance)
        self.assertTrue(np.all(h>0))

    def test_baseline_is_nested(self):
        r=np.array([1.,-2.,3.]); s=np.array([0.,-1.,1.])
        np.testing.assert_array_equal(variance_path([0,.2,.1,.8],r,initial_variance=2),
                                      variance_path([0,.2,.1,.8,0],r,s,2))

    def test_losses_and_degenerate_comparison(self):
        self.assertAlmostEqual(qlike_loss([0],[2])[0],np.log(2))
        with self.assertRaises(ValueError): qlike_loss([1],[0])
        stat,p,d=diebold_mariano(np.ones(20),np.ones(20))
        self.assertTrue(np.isnan(stat) and np.isnan(p)); self.assertEqual(d,0)
        np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])

    def test_core_imports_do_not_write_csv(self):
        from pathlib import Path
        before={str(p):p.stat().st_mtime_ns for p in Path('.').glob('*.csv')}
        for module in ['merge_data','forecast_comparision','yfinancedownload','fitting_garch11','run_sentiment','get_news']:
            importlib.import_module(module)
        after={str(p):p.stat().st_mtime_ns for p in Path('.').glob('*.csv')}
        self.assertEqual(before,after)


if __name__=='__main__':
    unittest.main(verbosity=2)
