using System;
using Accord.Statistics.Models.Regression;
using Accord.IO;
using Accord.MachineLearning;
using Accord.Statistics.Analysis;



namespace LogisticRegression
{
    public class Prediction
    {
        public int[] Answers
        {
            get => _answers;
        }
        public bool Success
        {
            get => _successfulModelLoad;
        }

        private int[] _answers;
        private bool _successfulModelLoad = true;
        private MultinomialLogisticRegression _mdl;
        public Prediction(in string _modelFname = "", MultinomialLogisticRegression model = null)
        {
            if (model == null) // Try to load a model from a file stream, if it fails return error
            {
                try
                {
                    MultinomialLogisticRegression mlr = Serializer.Load<MultinomialLogisticRegression>(_modelFname);

                }
                catch (Exception e)
                {
                    _successfulModelLoad = false;
                }

            }
            else
            {
                _mdl = model;
            }

        }

        public int[] Predict(in double[][] _testData)
        {
            return _answers = _mdl.Decide(_testData);
        }

        public ConfusionMatrix CreateConfusionMatrix(in int[] Expected, in int[] Predicted)
        {
            return new ConfusionMatrix(expected: Expected,
                predicted: Predicted);
        }
    }

}