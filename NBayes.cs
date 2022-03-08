using Accord.MachineLearning.Bayes;
using Accord.MachineLearning;
using Accord.Math;
using UtilityFuncs;
using System;
using Accord.Math.Optimization.Losses;
using Accord.IO;
using Accord.Statistics.Analysis;

namespace NBayes
{


    public class nBayes
    {
        // Fields
        public NaiveBayes BayesModel
        {
            get => _classifier;
        }
        public Accord.MachineLearning.Performance.CrossValidationResult<NaiveBayes, int[], int> CVResult
        {
            get => _result;
        }

        public double CrossValidationTrainingMean
        {
            get => _result.Training.Mean;
        }
        public double CrossValidationTrainingVariance
        {
            get => _result.Training.Variance;
        }


        private int[][] _inputMatrix;
        private int[] _labels;
        private NaiveBayes _classifier;
        private NaiveBayesLearning _learner;


        private Accord.MachineLearning.Performance.CrossValidationResult<NaiveBayes, int[], int> _result;
        // Methods

        /* Cross-validation is a technique for estimating the performance of a predictive model. 
         * It can be used to measure how the results of a statistical analysis will generalize to 
         * an independent data set.It is mainly used in settings where the goal is prediction, and
         * one wants to estimate how accurately a predictive model will perform in practice.
         *
         * One round of cross-validation involves partitioning a sample of data into complementary
         * subsets, performing the analysis on one subset (called the training set), and validating
         * the analysis on the other subset(called the validation set or testing set). To reduce
         * variability, multiple rounds of cross-validation are performed using different partitions,
         * and the validation results are averaged over the rounds
         *
         * Gets results based on performing a k-fold cross validation based on the input training set
         * Create a cross validation instance
         */
        public void crossValidation(in int Folds = 4)
        {
            var cv = CrossValidation.Create(k: Folds, learner: (p) => new NaiveBayesLearning(),
                    loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),
                    fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                    x: _inputMatrix, y: _labels);

            _result = cv.Learn(_inputMatrix, _labels);

        }

        // Constructor: Create a new Naive Bayes model
        public nBayes(in double[,] trainingSet, in int[] labelSet)
        {
            _inputMatrix = trainingSet.ToJagged().ToInt32();
            _labels = labelSet;
            // Create a new Naive Bayes learning instance
            _learner = new NaiveBayesLearning();

            // Create a Naive Bayes classifier and train with the input datasets
            //NaiveBayes classifier = learner.Learn( trainingSet.ToJagged().ToInt32(), labelSet
            _classifier = _learner.Learn(_inputMatrix, _labels);

        }

        public bool SaveModel(in string filename)
        {
            bool _success = true;
            try
            {
                _classifier.Save(filename, compression: SerializerCompression.None);
            }
            catch (Exception e)
            {
                _success = false;
            }
            return _success;
        }

    }

    public class Prediction
    {
        private NaiveBayes _model;
        public NaiveBayes Model
        {
            get => _model;
        }

        public Prediction(in string _modelname, in NaiveBayes _ModelObj = null) // Optional parameter??
        {
            if (_ModelObj == null)
            {
                _model = (Serializer.Load<NaiveBayes>(_modelname));
            }

        }
        public Prediction(in NaiveBayes _modelObject)
        {
            _model = _modelObject;
        }

        public int[] Predict(in double[,] DataMatrix)
        {
            int[,] inputs = DataMatrix.ToInt32();
            int[] answers = new int[inputs.Rows()];
            for (int i = 0; i < inputs.Rows(); i++)
            {
                answers[i] = _model.Decide(inputs.GetRow(i));
            }
            return answers;
        }

        public ConfusionMatrix CreateConfusionMatrix(in int[] Expected, in int[] Predicted)
        {
            return new ConfusionMatrix(expected: Expected, predicted: Predicted);

        }
    }
}