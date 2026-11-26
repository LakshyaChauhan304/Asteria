import unittest

from demo_app import models_disagree


class ModelDisagreementTests(unittest.TestCase):
    def test_three_predictions_are_not_all_equal(self):
        base = {"logistic_prediction": 1, "isolation_prediction": 1, "xgboost_prediction": 1}
        self.assertFalse(models_disagree(base))
        for model in base:
            changed = dict(base)
            changed[model] = 0
            self.assertTrue(models_disagree(changed))
        self.assertFalse(models_disagree(dict.fromkeys(base, 0)))
