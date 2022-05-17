using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math.Optimization; // L-BFGS
using Accord.MachineLearning; // needed for Cross Validation
using Accord.Statistics.Analysis; // For confusion matrices
using System;
using Accord.IO;
using Functions;


namespace LogisticRegression
{
    public class LRModel
    {
        /*
         * Public properties
         */

        public double Sensitivity //True Positives
        {
            get => _cm.Sensitivity;
        }

        public double Specificity //True Negatives
        {
            get => _cm.Specificity;
        }

        public double Precision
        {
            get => _cm.Precision;
        }

        public double Recall
        {
            get => _cm.Recall;
        }

        public double Fscore
        {
            get => _cm.FScore;
        }

        public double FalsePositives
        {
            get => _cm.FalsePositives;
        }

        public double FalseNegatives
        {
            get => _cm.FalseNegatives;
        }

        public MultinomialLogisticRegression modelObject
        {
            get => _mlr;
        }

        /*
         * Private properties
         */

        //private string _filename;
        private double[][] _trainingMatrix;
        //private int[] _labels;
        private MultinomialLogisticRegression _mlr;
        private GeneralConfusionMatrix _gcm;
        private ConfusionMatrix _cm;
        //private ConfusionMatrix _cv;
        public LRModel(in double[,] input, in int[] labels)  // Constructor
        {
            /* The L-BFGS algorithm is a member of the broad family of quasi-Newton optimization methods.
             * L-BFGS stands for 'Limited memory BFGS'. Indeed, L-BFGS uses a limited memory variation of
             * the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update to approximate the inverse Hessian matrix
             * (denoted by Hk). Unlike the original BFGS method which stores a dense approximation, L-BFGS
             * stores only a few vectors that represent the approximation implicitly. Due to its moderate
             * memory requirement, L-BFGS method is particularly well suited for optimization problems with
             * a large number of variables. 
             */
            double[][] _jaggedInput = Util_Methods.convertToJaggedArray(input);

            // Create a LBFGS model
            _trainingMatrix = _jaggedInput;

            var mlbfgs = new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>();

            // Estimate using the data against a logistic regression
            _mlr = mlbfgs.Learn(_trainingMatrix, labels);

            _cm = ConfusionMatrix.Estimate(_mlr, _trainingMatrix, labels);


        }

        public void CrossValidate(in int _folds, in double[][] _trainingMatrix, in int[] _labels)
        {
            // Create a cross validation model derived from the training set to measure the performance of this
            // predictive model and estimate how well we expect the model will generalize. The algorithm executes
            // multiple rounds of cross validation on different partitions and averages the results. 
            //

            var cv = CrossValidation.Create(k: _folds, learner: (p) => new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>(),
                loss: (actual, expected, p) => new Accord.Math.Optimization.Losses.ZeroOneLoss(expected).Loss(actual),
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                x: _trainingMatrix, y: _labels);
            var _cv = cv.Learn(_trainingMatrix, _labels);
        }

        public bool SaveModel(in string _fName)
        {
            bool _success = true;
            try
            {
                _mlr.Save(_fName, compression: SerializerCompression.None);
            }
            catch (Exception e)
            {
                _success = false;
            }
            return _success;
        }

        public ConfusionMatrix CreateConfusionMatrix(in int[] Expected, in int[] Predicted)
        {
            return new ConfusionMatrix(expected: Expected,
                predicted: Predicted);
        }
    }

}
