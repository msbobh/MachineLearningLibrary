namespace resources
{
    class strings
    {
        static public string usage = "NNTrain <training file> <label file> <opt> model file name " +
            "Outputs ==> model file";
        static public string SampleMisMatch = "Sample count between training file and label file does not match: {0}, {1}";
        //static public string TrResults = "  Results of Training run:";
        static public string FalsePos = "     False Positives: ";
        static public string FalseNeg = "     False Negatives: ";
        static public string Fscore = "     Fscore: ";
        static public string StartingUp = " Neural Network (Accord.net) Training Utility Starting...\n";
        static public string Predictions = "Prediction accuracy of training set = ";

    }
}