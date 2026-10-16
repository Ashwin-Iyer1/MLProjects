import unittest
import numpy as np
import pandas as pd
from ml_lab.titanic import ROOT, prepare_features, make_pipeline, build_report
from ml_lab.images import prepare_mnist
from ml_lab.crashes import summarize_crashes


class TitanicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_holdout_and_training_ids_disjoint_and_complete(self):
        split = self.report['split']
        self.assertFalse(set(split['trainIds']) & set(split['holdoutIds']))
        self.assertEqual(len(split['trainIds']) + len(split['holdoutIds']), 891)
        self.assertEqual(len(self.report['predictions']), 179)

    def test_metrics_reconcile_with_saved_predictions(self):
        r = self.report
        accuracy = sum((p['probability'] >= .5) == p['actual'] for p in r['predictions']) / len(r['predictions'])
        self.assertAlmostEqual(accuracy, r['metrics']['accuracy'])
        self.assertEqual(sum(map(sum, r['metrics']['confusion'])), 179)

    def test_holdout_values_cannot_change_training_imputer(self):
        frame = pd.read_csv(ROOT/'TitanicPred/titanic/train.csv')
        train = frame[frame.PassengerId.isin(self.report['split']['trainIds'])]
        self.assertAlmostEqual(train.Age.median(), self.report['model']['trainingMedians']['Age'])
        self.assertAlmostEqual(train.Fare.median(), self.report['model']['trainingMedians']['Fare'])

    def test_unknown_categories_keep_prediction_shape(self):
        frame = pd.read_csv(ROOT/'TitanicPred/titanic/train.csv').head(100)
        x = prepare_features(frame)
        model = make_pipeline().fit(x, frame.Survived)
        future = x.iloc[:1].copy(); future['Embarked'] = 'UNKNOWN'
        self.assertEqual(model.predict_proba(future).shape, (1, 2))

    def test_duplicate_passenger_ids_rejected(self):
        frame = pd.read_csv(ROOT/'TitanicPred/titanic/train.csv').head(2)
        frame['PassengerId'] = 1
        with self.assertRaises(ValueError): prepare_features(frame)


class ImageTests(unittest.TestCase):
    def test_single_and_batch_paths_match(self):
        a = np.arange(784).reshape(28, 28) % 256
        np.testing.assert_array_equal(prepare_mnist(a), prepare_mnist(a[None]))

    def test_original_axis_one_l2_contract(self):
        a = np.full((2, 28, 28), 255)
        np.testing.assert_allclose(prepare_mnist(a), 1/np.sqrt(28), rtol=1e-6)

    def test_black_pixels_remain_finite_zero(self):
        a = prepare_mnist(np.zeros((28, 28)))
        self.assertTrue(np.isfinite(a).all()); self.assertEqual(a.sum(), 0)

    def test_inversion_and_invalid_shapes(self):
        self.assertEqual(prepare_mnist(np.full((28,28),255), invert=True).sum(), 0)
        for a in [np.zeros((27,28)), np.full((28,28),np.nan),np.full((28,28),256)]:
            with self.assertRaises(ValueError): prepare_mnist(a)


class CrashTests(unittest.TestCase):
    def fixtures(self):
        return (pd.DataFrame({'CRASH_RECORD_ID':['a'],'INJURIES_TOTAL':[2]}),
                pd.DataFrame({'CRASH_RECORD_ID':['a','a'],'CRASH_UNIT_ID':[1,2],'MAKE':['X','X']}),
                pd.DataFrame({'CRASH_RECORD_ID':['a','a','a'],'PERSON_ID':[1,2,3],'AGE':[20,30,-1]}))

    def test_many_to_many_join_does_not_multiply_crashes_or_injuries(self):
        result=summarize_crashes(*self.fixtures())
        self.assertEqual((result['crashCount'],result['vehicleCount'],result['personCount'],result['knownInjuries']),(1,2,3,2))
        self.assertEqual(result['makes'].iloc[0].unique_crashes,1)
        self.assertEqual(result['excludedAges'],1)

    def test_duplicates_and_orphans_fail(self):
        c,v,p=self.fixtures();v['CRASH_UNIT_ID']=1
        with self.assertRaises(ValueError):summarize_crashes(c,v,p)
        c,v,p=self.fixtures();p.loc[0,'CRASH_RECORD_ID']='missing'
        with self.assertRaises(ValueError):summarize_crashes(c,v,p)

    def test_source_age_is_not_mutated_to_birth_year(self):
        c,v,p=self.fixtures();original=p.copy(deep=True);summarize_crashes(c,v,p);pd.testing.assert_frame_equal(p,original)


if __name__=='__main__': unittest.main()
