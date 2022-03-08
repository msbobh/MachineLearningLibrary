using Accord.Neuro.Networks;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System;
using Accord.Statistics.Analysis;
using UtilityFuncs;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.Performance;

namespace DeepBelief
{
    public class DBNetworkModel
    {
        public DeepBeliefNetwork NetworkObject
        {
            get => _netWorkObj;
        }

        public int Epochs
        {
            get => _epochs;
            set
            {
                if (value <= 0)
                    _epochs = 5000;
                else
                    _epochs = value;
            }
        }

        public int Inputs
        {
            get => _inputs;
        }

        public int Outputs
        {
            get => _outputs;
        }

        public int Layers
        {
            get => _hiddenLayers;
        }

        // Private Variables
        private DeepBeliefNetwork _netWorkObj;
        private int _epochs;
        private int _inputs;
        private int _hiddenLayers;
        private int _outputs;



        /*
         * Neural Network Constructor Requires the following:
         *  Number of inputs
         *  Number of hidden layers
         *  Number of outputs
         * 
         */

        public DBNetworkModel(in int NumInputs, in int hiddenLayers, in int numOutputs, in double[,] inputs,
            in double[,] labels)
        {
            /*
             * Deep belief: multiple layers of hidden units connected as a bipartite graph. The
             * hidden units are trained on a set of examples without supervision in preparation for
             * supervised training on the input set.
             * 
             * Each of the sub layers are trained using contrastive divergence in turn 
             */

            double[][] _jaggedInputs = externalFunc.convertToJaggedArray(inputs);
            double[][] _jaggedLabels = externalFunc.convertToJaggedArray(labels);
            _netWorkObj = new
                DeepBeliefNetwork(NumInputs, hiddenLayers, numOutputs);
            // Network weights must be initialized but can't use 0, preferable to use randomized initializers
            // Accord provides an easy way to use Gaussian weights
            new GaussianWeights(_netWorkObj, 0.1).Randomize();

            // Update the visible layer with the new weights
            _netWorkObj.UpdateVisibleWeights();


            // Setup the learning algorithm.
            DeepBeliefNetworkLearning _teacher = new DeepBeliefNetworkLearning(_netWorkObj)
            {
                Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(hidden: h, visible: v)
                {
                    LearningRate = 0.1,
                    Momentum = 0.5,
                    Decay = 0.001,
                }
            };

            // Unsupervised learning on each hidden layer, except for the output.
            for (int i = 0; i < _netWorkObj.Layers.Length - 1; i++)
            {
                _teacher.LayerIndex = i;

                // Compute the learning data  should be used
                var layerInput = _teacher.GetLayerInput(_jaggedInputs);

                // Train the layer iteratively
                for (int j = 0; j < 5000; j++)
                    _teacher.RunEpoch(layerInput);
            }

            // Supervised learning on entire network, to provide output classification.
            var backpropagation = new BackPropagationLearning(_netWorkObj)
            {
                LearningRate = 0.1,
                Momentum = 0.5
            };

            // Run supervised learning.
            for (int i = 0; i < Epochs; i++)
                backpropagation.RunEpoch(_jaggedInputs, _jaggedLabels);

            _inputs = _netWorkObj.InputsCount;
            _outputs = _netWorkObj.OutputCount;
            _hiddenLayers = hiddenLayers;
        }

        public bool SaveModel(in string fName)
        {
            bool _success = true;
            try
            {
                _netWorkObj.Save(fName);
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

        public void CrossValidate(in int _folds, in int [] _trainingMatrix, in int[] _labels)
        {
            var crossvalidation = new CrossValidation<SupportVectorMachine<Accord.Statistics.Kernels.Linear, double[]>, double[]>()
            {
                K = 3, // Use 3 folds in cross-validation

                // Indicate how learning algorithms for the models should be created
                Learner = (s) => new Accord.MachineLearning.VectorMachines.Learning.SequentialMinimalOptimization<Accord.Statistics.Kernels.Linear, double[]>()
                {
                    Complexity = 100
                },                                                

                // Indicate how the performance of those models will be measured
                Loss = (expected, actual, p) => new Accord.Math.Optimization.Losses.ZeroOneLoss(expected).Loss(actual),

                Stratify = false, // do not force balancing of classes
            };
            var foo = CrossValidation.Create (k: _folds, learner: (l) => new DeepBeliefNetworkLearning(),loss: (actual, expected, l)=> ;
            /*var cv = Accord.MachineLearning.CrossValidation.Create(k: _folds, learner: (p) => new DeepBeliefNetworkLearning<>,
                loss: (actual, expected, p) => new Accord.Math.Optimization.Losses.ZeroOneLoss(expected).Loss(actual),
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                x: _trainingMatrix, y: _labels);
            var _cv = cv.Learn(_trainingMatrix, _labels);*/

        }


    }

    public class DBPrediction
    {
        private DeepBeliefNetwork _netWorkObj;

        public DBPrediction(in DeepBeliefNetwork _modelObj)
        {
            _netWorkObj = _modelObj;

        }
        public int[] Predict(in double[,] input)
        {
            // The compute method needs a jagged array as input
            double[][] jaggedInput = UtilityFuncs.externalFunc.convertToJaggedArray(input);
            double[][] predicted = new double[jaggedInput.Length][];

            for (int i = 0; i <= jaggedInput.Length - 1; i++)
            {
                predicted[i] = _netWorkObj.Compute(jaggedInput[i]);
            }
            int[] intPredicted = new int[predicted.Length];
            // Set the threshold for true/false at 0.5
            for (int i = 0; i <= predicted.Length - 1; i++)
            {
                if (predicted[i][0] < 0.5)
                    intPredicted[i] = 0;
                else
                    intPredicted[i] = 1;
            }
            return intPredicted;
        }
    }

}